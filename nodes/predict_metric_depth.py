"""SharpPredictMetricDepth — image -> disparity-head metric depth only.

Tap SHARP's encoder + disparity head; skip the gaussian decoder entirely.
Fastest path to a clean per-pixel depth map at SHARP's native 1536²
resolution, paired with extrinsics_mdepth (pass-through) and
intrinsics_mdepth (pixel-K rescaled to 1536). The triplet wires straight
into SharpDepthMerge with no K/depth-shape mismatch.

Why separate from SharpPredictGaussianAttrs:
  - ~2× faster per face (no init_model + feature_model + prediction_head
    forward passes).
  - Lower VRAM (skip decoder allocs).
  - Cleaner workflows where you only want depth (mesh build, navmesh,
    downstream depth-aligned gaussian prediction, etc.).

Shares `_encode_cache` with SharpPredictGaussianAttrs so encoding the
same image across both nodes is a single forward pass.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .predict_gaussian_attrs import (
    _compute_image_hash, _monodepth_to, _encode_cache,
)
from .utils.image import convert_focallength


def _p(msg: str) -> None:
    print(f"[SharpPredictMetricDepth] {msg}", file=sys.stderr, flush=True)


def _rescale_K(intr_in, target_w, target_h, src_w, src_h):
    sx = float(target_w) / float(src_w)
    sy = float(target_h) / float(src_h)
    k = intr_in.detach().clone().float() if isinstance(intr_in, torch.Tensor) \
        else torch.as_tensor(np.asarray(intr_in)).float().clone()
    if k.dim() == 2:
        k[0, :] *= sx
        k[1, :] *= sy
        k_dbg = k
    elif k.dim() == 3:
        k[:, 0, :] *= sx
        k[:, 1, :] *= sy
        k_dbg = k[0]
    else:
        raise ValueError(f"intrinsics must be (3,3) or (N,3,3), got {tuple(k.shape)}")
    return k, float(k_dbg[0, 0])


class SharpPredictMetricDepth(io.ComfyNode):
    """SHARP encoder + disparity head only -> per-pixel metric depth at 1536²."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharpPredictMetricDepth",
            display_name="SHARP Predict Metric Depth",
            category="SHARP",
            description=(
                "Run only SHARP's encoder + disparity head (skip the gaussian "
                "decoder). Outputs per-pixel metric depth at native 1536² + "
                "paired extrinsics/intrinsics. ~2× faster than "
                "SharpPredictGaussianAttrs when you only want depth. Wire "
                "into SharpDepthMerge for clean equirect depth merge."
            ),
            inputs=[
                io.Custom("SHARP_MODEL_CONFIG").Input("model"),
                io.Image.Input("image"),
                io.Float.Input(
                    "focal_length_mm", default=30.0, min=0.0, max=500.0,
                    step=0.1, optional=True,
                    tooltip="Focal length in mm (35mm equiv). Ignored if "
                            "intrinsics provided."),
                io.Custom("EXTRINSICS").Input(
                    "extrinsics", optional=True,
                    tooltip="Pass-through to extrinsics_mdepth output."),
                io.Custom("INTRINSICS").Input(
                    "intrinsics", optional=True,
                    tooltip="If provided, overrides focal_length_mm. "
                            "Re-emitted on intrinsics_mdepth rescaled to 1536²."),
            ],
            outputs=[
                io.Image.Output(
                    display_name="layer_0_metric_depth",
                    tooltip="[B, 1536, 1536, 3] LAYER-0 (front/visible) metric "
                            "depth from SHARP's disparity head, native "
                            "resolution. Same tensor that feeds the gaussian "
                            "decoder's layer-0 base values at inference."),
                io.Image.Output(
                    display_name="layer_1_metric_depth",
                    tooltip="[B, 1536, 1536, 3] LAYER-1 (back/occluded) metric "
                            "depth — SHARP's hallucinated backplate surface. "
                            "Typically tracks layer 0 in flat regions and "
                            "diverges at occlusion boundaries (column edges -> "
                            "sky depth behind, etc.). Feeds layer-1 gaussian "
                            "base values."),
                io.Custom("EXTRINSICS").Output(
                    display_name="extrinsics_mdepth",
                    tooltip="Pass-through of input extrinsics (resolution-"
                            "independent). Applies to both depth layers."),
                io.Custom("INTRINSICS").Output(
                    display_name="intrinsics_mdepth",
                    tooltip="Intrinsics rescaled to the 1536² depth grid. "
                            "Applies to both depth layers."),
                io.Image.Output(
                    display_name="layer_0_points_raw",
                    tooltip="[B, 1536, 1536, 3] LAYER-0 per-pixel 3D point "
                            "map in CAMERA SPACE — (X, Y, Z) in meters at "
                            "each pixel. Wire directly into "
                            "PanoramaDepthMerge.face_points (which expects "
                            "this exact shape — NOT the scalar depth)."),
                io.Image.Output(
                    display_name="layer_1_points_raw",
                    tooltip="[B, 1536, 1536, 3] LAYER-1 per-pixel 3D point "
                            "map in camera space."),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        model,
        image: torch.Tensor,
        focal_length_mm: float = 30.0,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ):
        global _encode_cache
        import comfy.model_management
        from .load_model import _load_sharp_model

        patcher = _load_sharp_model(model)
        predictor = patcher.model
        device = patcher.load_device

        if image.dim() == 3:
            image = image.unsqueeze(0)
        B = image.shape[0]
        t_start = time.time()

        # Auto-construct camera defaults when inputs are None so the output
        # `extrinsics_mdepth` / `intrinsics_mdepth` sockets emit real
        # tensors. Downstream consumers (SharpDepthMerge) crash on
        # `np.asarray(None)`; this matches the same fix applied to
        # SharpPredictGaussianAttrs. image is [B, H, W, 3].
        _img_H, _img_W = int(image.shape[1]), int(image.shape[2])
        if extrinsics is None:
            extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
        if intrinsics is None:
            _f_px_default = float(convert_focallength(
                _img_W, _img_H, max(0.1, float(focal_length_mm or 30.0)),
            ))
            _K_default = torch.tensor(
                [
                    [_f_px_default, 0.0,           _img_W / 2.0],
                    [0.0,           _f_px_default, _img_H / 2.0],
                    [0.0,           0.0,           1.0],
                ],
                dtype=torch.float32,
            )
            intrinsics = _K_default.unsqueeze(0).repeat(B, 1, 1)
            _p(
                f"intrinsics not wired -> using identity-style K "
                f"(focal={_f_px_default:.1f}px @ image {_img_W}×{_img_H}, "
                f"35mm-equiv {focal_length_mm:.1f}mm). Pass intrinsics from "
                f"PanoramaSplit for per-face accurate K."
            )
        else:
            # Convention sniff: PanoPack's PanoramaSplit emits NORMALIZED K
            # (fx~=0.5, cx~=0.5 for 90° fov via utils3d.np.intrinsics_from_fov,
            # units in [0,1]), whereas Sharp's predict path assumes pixel-K
            # (fx in the hundreds). Without this conversion, `f_px = K[0,0] *
            # (1536/width)` resolves to 0.5 instead of ~768, the disparity ->
            # depth math collapses, and every face comes out as a blank (~0)
            # depth map. Detect normalized (fx < 2.0) and rescale once to
            # pixel-K for the face image's (width, height). All downstream
            # uses (f_px, _rescale_K to 1536) then see pixel-K.
            _sample_fx = float(intrinsics[0, 0, 0] if intrinsics.dim() == 3 else intrinsics[0, 0])
            if _sample_fx < 2.0:
                intrinsics = intrinsics.clone().float()
                if intrinsics.dim() == 3:
                    intrinsics[:, 0, :] *= float(_img_W)
                    intrinsics[:, 1, :] *= float(_img_H)
                else:
                    intrinsics[0, :] *= float(_img_W)
                    intrinsics[1, :] *= float(_img_H)
                _p(
                    f"detected normalized intrinsics (fx<2); rescaled to "
                    f"pixel-K for {_img_W}×{_img_H}: fx={float(intrinsics[0, 0, 0] if intrinsics.dim() == 3 else intrinsics[0, 0]):.1f}"
                )

        internal_shape = (1536, 1536)
        input_shape = [1, 3, internal_shape[0], internal_shape[1]]
        memory_required = patcher.memory_required(input_shape)
        comfy.model_management.load_models_gpu(
            [patcher], memory_required=memory_required,
        )

        metric_depths_l0 = []  # front surface
        metric_depths_l1 = []  # back/occluded surface
        last_width = last_height = None
        for b in range(B):
            img_np = image[b].cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image[b])
            if img_np.dtype != np.uint8:
                img_np = (np.clip(img_np, 0, 1) * 255 + 0.5).astype(np.uint8)
            height, width = img_np.shape[:2]
            last_width, last_height = width, height
            image_hash = _compute_image_hash(img_np)

            if _encode_cache["image_hash"] == image_hash:
                monodepth_output = _monodepth_to(_encode_cache["monodepth_output"], device)
            else:
                _encode_cache["image_hash"] = None
                image_pt = (
                    torch.from_numpy(img_np.copy()).float().to(device).permute(2, 0, 1) / 255.0
                )
                image_resized_pt = F.interpolate(
                    image_pt[None],
                    size=(internal_shape[1], internal_shape[0]),
                    mode="bilinear", align_corners=True,
                )
                monodepth_output, _ = predictor.encode(image_resized_pt)
                _encode_cache["image_hash"] = image_hash
                _encode_cache["monodepth_output"] = _monodepth_to(monodepth_output, "cpu")
                _encode_cache["image_resized"] = image_resized_pt.cpu()
                _encode_cache["original_shape"] = (height, width)
                comfy.model_management.soft_empty_cache()

            if intrinsics is not None:
                intr_b = intrinsics[b] if intrinsics.dim() == 3 else intrinsics
                f_px = float(intr_b[0, 0]) * (internal_shape[0] / width)
            else:
                # Match SharpPredict's 35mm-equivalent diagonal formula
                # (`convert_focallength`) so the depth here is byte-identical
                # to what `predictor.decode` sees inside SharpPredict for the
                # same image. Previously this used (width/36)·f_mm which
                # disagreed by a factor of ~1.18× for square inputs.
                f_px = float(convert_focallength(width, height, max(0.1, float(focal_length_mm or 30.0))))

            disparity_factor_scalar = f_px / width
            monodepth_disparity = monodepth_output.disparity  # [1, 2, 1536, 1536]
            metric_depth_full = (
                disparity_factor_scalar / monodepth_disparity.clamp(min=1e-4)
            )  # [1, 2, 1536, 1536]
            # Layer 0 = front (max disparity post-sort), layer 1 = back.
            metric_depths_l0.append(metric_depth_full[0, 0].cpu())  # [1536, 1536]
            if metric_depth_full.shape[1] >= 2:
                metric_depths_l1.append(metric_depth_full[0, 1].cpu())
            else:
                metric_depths_l1.append(metric_depth_full[0, 0].cpu().clone())

        metric_batch_l0 = torch.stack(metric_depths_l0, dim=0)  # [B, 1536, 1536]
        metric_batch_l1 = torch.stack(metric_depths_l1, dim=0)  # [B, 1536, 1536]
        metric_img_l0 = metric_batch_l0.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
        metric_img_l1 = metric_batch_l1.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()

        # Per-face 3D point maps in camera space:
        #   x_cam = (u - cx) / fx
        #   y_cam = (v - cy) / fy
        #   points[u, v] = (x_cam · Z, y_cam · Z, Z)
        # By construction ||points|| == ray distance (Euclidean from camera
        # center), which is what PanoramaDepthMerge.face_points consumes.
        # Built at 1536² to match the depth grid; `intrinsics` is already in
        # pixel-K convention here (the fx<2 normalized sniff at the top of
        # execute() rescaled if PanoPack-style normalized K was passed).
        Hd = Wd = internal_shape[0]
        grid_device = metric_batch_l0.device
        points_l0 = []
        points_l1 = []
        for b in range(B):
            intr_b = intrinsics[b] if intrinsics.dim() == 3 else intrinsics
            sx = Wd / float(last_width)
            sy = Hd / float(last_height)
            fx_d = float(intr_b[0, 0]) * sx
            fy_d = float(intr_b[1, 1]) * sy
            cx_d = float(intr_b[0, 2]) * sx
            cy_d = float(intr_b[1, 2]) * sy
            uu = torch.arange(Wd, dtype=torch.float32, device=grid_device)
            vv = torch.arange(Hd, dtype=torch.float32, device=grid_device)
            uu_g, vv_g = torch.meshgrid(uu, vv, indexing="xy")  # (Hd, Wd)
            x_ndc = (uu_g - cx_d) / fx_d
            y_ndc = (vv_g - cy_d) / fy_d
            Z0 = metric_batch_l0[b]
            Z1 = metric_batch_l1[b]
            points_l0.append(torch.stack([x_ndc * Z0, y_ndc * Z0, Z0], dim=-1))
            points_l1.append(torch.stack([x_ndc * Z1, y_ndc * Z1, Z1], dim=-1))
        points_batch_l0 = torch.stack(points_l0, dim=0).contiguous()  # (B, Hd, Wd, 3)
        points_batch_l1 = torch.stack(points_l1, dim=0).contiguous()

        if intrinsics is not None:
            intrinsics_mdepth_out, k_fx_md_dbg = _rescale_K(
                intrinsics, internal_shape[1], internal_shape[0],
                last_width, last_height,
            )
        else:
            intrinsics_mdepth_out = None
            k_fx_md_dbg = None
        extrinsics_mdepth_out = extrinsics

        m_med_l0 = float(metric_batch_l0.median())
        m_med_l1 = float(metric_batch_l1.median())
        # ||points|| range as a sanity check that depth × unprojection
        # produced sensible scene-scale 3D positions.
        pts_norm = points_batch_l0.norm(dim=-1)
        pts_min = float(pts_norm.min())
        pts_max = float(pts_norm.max())
        elapsed = time.time() - t_start
        k_str = f", K_{internal_shape[0]} fx->{k_fx_md_dbg:.1f}" if k_fx_md_dbg is not None else ""
        _p(
            f"{B} face(s) @ {internal_shape[0]}²; "
            f"layer0/layer1 depth median={m_med_l0:.2f}/{m_med_l1:.2f}m{k_str}; "
            f"points_raw ||v||={pts_min:.2f}–{pts_max:.2f}m; {elapsed:.1f}s"
        )

        return io.NodeOutput(
            metric_img_l0, metric_img_l1,
            extrinsics_mdepth_out, intrinsics_mdepth_out,
            points_batch_l0, points_batch_l1,
        )


NODE_CLASS_MAPPINGS = {"SharpPredictMetricDepth": SharpPredictMetricDepth}
NODE_DISPLAY_NAME_MAPPINGS = {"SharpPredictMetricDepth": "SHARP Predict Metric Depth"}
