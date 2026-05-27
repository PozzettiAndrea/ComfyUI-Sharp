"""SharpPredictGaussianAttrs node for ComfyUI-Sharp.

Runs SHARP inference and exposes the per-pixel gaussian decoder output as
structured per-layer tensors instead of flattening to a PLY.

Why: SHARP outputs `num_layers = 2` gaussians per stride-pixel (layer 0 =
visible/near surface, layer 1 = back/occluded surface). The gaussian
composer's pre-flatten shape is:

    mean_vectors    [B, 3, num_layers, H_grid, W_grid]
    singular_values [B, 3, num_layers, H_grid, W_grid]
    quaternions     [B, 4, num_layers, H_grid, W_grid]
    colors          [B, 3, num_layers, H_grid, W_grid]
    opacities       [B,    num_layers, H_grid, W_grid]

Where H_grid = W_grid = `predictor.output_resolution` (typically 768 when
SHARP internal_resolution = 1536). The `gaussian_composer` then flattens
to [B, num_layers*H*W, C] for save_ply; the post-flatten order is
contiguous in (layer, h, w) — layer 0 first, layer 1 second.

This node un-flattens that, separates layers, and emits:

    first_layer_depth_refined   IMAGE [B, H, W, 3]  -- ndc-z of layer 0
    first_layer_gaussian_attrs  MULTIBAND_IMAGE [B, 14, H, W]
    second_layer_depth_refined  IMAGE [B, H, W, 3]  -- ndc-z of layer 1
    second_layer_gaussian_attrs MULTIBAND_IMAGE [B, 14, H, W]

The "refined depth" is the gaussian's ndc-z. Since
`ndc_matrix[2, 2] = 1.0` in `unproject_gaussians`, ndc-z equals
camera-space z. That's the post-decoder refined depth — tighter than
the disparity head's pre-refinement output exposed by SharpPredictDepth.

Multiband channel order (per layer):

    0:  position_x   (ndc, in [-1, 1])
    1:  position_y   (ndc, in [-1, 1])
    2:  position_z   (ndc / camera z, same as depth_refined output;
                       duplicated here so multiband is self-contained)
    3:  scale_x      (Σ-singular value)
    4:  scale_y
    5:  scale_z
    6:  quaternion_w
    7:  quaternion_x
    8:  quaternion_y
    9:  quaternion_z
    10: color_r      (linear or SH-DC; see gaussians.py)
    11: color_g
    12: color_b
    13: opacity      (post-activation, in [0, 1])

Downstream: feed `first_layer_depth_refined` (across N faces) into
AlignDepthMaps to recover per-face scales, then multiply the scales into
each multiband's position channels before reconstructing a PLY.
"""

from __future__ import annotations

import hashlib
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from comfy_api.latest import io

log = logging.getLogger("sharp")


# Channel order for the multiband output. Mirrors the gaussian composer's
# concat order in the un-flattened tensor we assemble.
GAUSS_ATTR_CHANNEL_NAMES = [
    "position_x", "position_y", "position_z",
    "scale_x", "scale_y", "scale_z",
    "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z",
    "color_r", "color_g", "color_b",
    "opacity",
]
NUM_GAUSS_ATTR_CHANNELS = len(GAUSS_ATTR_CHANNEL_NAMES)  # 14


# Same encode cache as SharpPredict — feature-extraction is the heaviest
# step, and a user iterating on focal length should hit the cache.
_encode_cache: dict[str, object] = {
    "image_hash": None,
    "monodepth_output": None,
    "image_resized": None,
    "original_shape": None,
}


def _compute_image_hash(image_np: np.ndarray) -> str:
    return hashlib.sha256(image_np.tobytes()).hexdigest()[:16]


def _monodepth_to(output, device):
    """Move all tensors in a MonodepthOutput to the given device."""
    from .sharp.model import MonodepthOutput
    return MonodepthOutput(
        disparity=output.disparity.to(device),
        encoder_features=[t.to(device) for t in output.encoder_features],
        decoder_features=output.decoder_features.to(device),
        output_features=[t.to(device) for t in output.output_features],
        intermediate_features=[t.to(device) for t in output.intermediate_features],
    )


def _split_layers_from_flat_gaussians(
    gaussians,
    num_layers: int,
    H_grid: int,
    W_grid: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten Gaussians3D -> per-layer per-pixel attribute stack.

    The composer's flatten was:
        x.permute(0, 2, 3, 4, 1).flatten(1, 3)
    starting from [B, C, num_layers, H, W]. The flatten dim ordering is
    (num_layers -> H -> W), so reshaping back to [B, num_layers, H, W, C]
    recovers the original grid layout.

    Returns:
        layer0_attrs:  [B, H, W, 14]
        layer1_attrs:  [B, H, W, 14]   (for num_layers=2; if num_layers
                                        differs, raises)
    """
    mean = gaussians.mean_vectors            # [B, N, 3]
    scale = gaussians.singular_values        # [B, N, 3]
    quat = gaussians.quaternions             # [B, N, 4]
    color = gaussians.colors                 # [B, N, 3]
    opac = gaussians.opacities               # [B, N]

    B, N, _ = mean.shape
    expected_N = num_layers * H_grid * W_grid
    if N != expected_N:
        raise ValueError(
            f"gaussian count mismatch: got N={N}, expected "
            f"num_layers={num_layers} * H={H_grid} * W={W_grid} = {expected_N}. "
            f"Check predictor.output_resolution / num_layers."
        )

    # Un-flatten: [B, N, C] -> [B, num_layers, H, W, C]
    mean5 = mean.view(B, num_layers, H_grid, W_grid, 3)
    scale5 = scale.view(B, num_layers, H_grid, W_grid, 3)
    quat5 = quat.view(B, num_layers, H_grid, W_grid, 4)
    color5 = color.view(B, num_layers, H_grid, W_grid, 3)
    opac5 = opac.view(B, num_layers, H_grid, W_grid, 1)

    # Concat into a single [B, num_layers, H, W, 14] tensor in the order
    # the multiband channel names declare.
    stacked = torch.cat([mean5, scale5, quat5, color5, opac5], dim=-1)
    if stacked.shape[-1] != NUM_GAUSS_ATTR_CHANNELS:
        raise RuntimeError(
            f"channel count mismatch: stacked={stacked.shape[-1]}, "
            f"expected {NUM_GAUSS_ATTR_CHANNELS}"
        )
    if num_layers < 2:
        raise ValueError(
            f"expected num_layers>=2 for SHARP's two-layer output, got {num_layers}"
        )
    return stacked[:, 0], stacked[:, 1]


def _build_multiband(attrs: torch.Tensor) -> dict:
    """Convert [B, H, W, 14] -> MULTIBAND_IMAGE dict ([B, 14, H, W])."""
    if attrs.dim() != 4 or attrs.shape[-1] != NUM_GAUSS_ATTR_CHANNELS:
        raise ValueError(
            f"expected [B, H, W, {NUM_GAUSS_ATTR_CHANNELS}], got {tuple(attrs.shape)}"
        )
    samples = attrs.permute(0, 3, 1, 2).contiguous().float()  # [B, C, H, W]
    return {
        "samples": samples,
        "channel_names": list(GAUSS_ATTR_CHANNEL_NAMES),
        "metadata": {
            "source": "SharpPredictGaussianAttrs",
            "ndc_space": True,
            "note": (
                "position channels are in NDC space (ndc_matrix[2,2]=1.0 "
                "so position_z == camera-space z). Apply per-face scale "
                "to positions and run unproject_gaussians to recover "
                "world coords."
            ),
        },
    }


class SharpPredictGaussianAttrs(io.ComfyNode):
    """SHARP inference -> per-layer depth + multiband gaussian attributes."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharpPredictGaussianAttrs",
            display_name="SHARP Predict (Depth + Gaussian Attrs)",
            category="SHARP",
            description=(
                "Run SHARP inference and expose the per-pixel gaussian "
                "decoder output as structured per-layer tensors (depth + "
                "MULTIBAND_IMAGE of all gaussian attributes) instead of "
                "flattening to a PLY file.\n\n"
                "Use the depth outputs to drive AlignDepthMaps for per-"
                "face scale recovery, then multiply scales into the "
                "multiband position channels before reconstructing PLYs."
            ),
            inputs=[
                io.Custom("SHARP_MODEL_CONFIG").Input("model"),
                io.Image.Input("image"),
                io.Float.Input(
                    "focal_length_mm", default=30.0, min=0.0, max=500.0,
                    step=0.1, optional=True,
                    tooltip="Focal length in mm (35mm equiv). 0 = 30mm. "
                            "Ignored if intrinsics provided."),
                io.Custom("EXTRINSICS").Input(
                    "extrinsics", optional=True,
                    tooltip="Pass-through (e.g. from SamplePanorama / "
                            "SharpPanoramaCubeSplit). Re-emitted on the "
                            "extrinsics output so downstream nodes "
                            "(AlignDepthMaps) can wire them."),
                io.Custom("INTRINSICS").Input(
                    "intrinsics", optional=True,
                    tooltip="Pass-through. If provided, overrides "
                            "focal_length_mm. Re-emitted on the intrinsics "
                            "output."),
            ],
            outputs=[
                io.Image.Output(
                    display_name="first_layer_depth_refined",
                    tooltip="[B, H, W, 3] depth broadcast across 3 channels. "
                            "Layer-0 (visible surface) post-decoder z."),
                io.Custom("MULTIBAND_IMAGE").Output(
                    display_name="first_layer_gaussian_attrs",
                    tooltip="[B, 14, H, W] all layer-0 gaussian attributes. "
                            "Channels: position_xyz, scale_xyz, quat_wxyz, "
                            "color_rgb, opacity."),
                io.Image.Output(
                    display_name="second_layer_depth_refined",
                    tooltip="[B, H, W, 3] layer-1 (back/occluded surface) "
                            "post-decoder z."),
                io.Custom("MULTIBAND_IMAGE").Output(
                    display_name="second_layer_gaussian_attrs",
                    tooltip="[B, 14, H, W] all layer-1 gaussian attributes."),
                io.Custom("EXTRINSICS").Output(display_name="extrinsics"),
                io.Custom("INTRINSICS").Output(display_name="intrinsics"),
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

        log.debug(
            f"[SharpPredictGaussianAttrs] processing {B} image(s); "
            f"output_resolution={predictor.output_resolution}, "
            f"num_layers=2 (SHARP convention)"
        )
        t_start = time.time()

        # Process per-image (matches existing SharpPredict structure).
        first_depths = []
        first_attrs = []
        second_depths = []
        second_attrs = []

        # Load model to GPU with the budget the existing predict path uses.
        internal_shape = (1536, 1536)
        input_shape = [1, 3, internal_shape[0], internal_shape[1]]
        memory_required = patcher.memory_required(input_shape)
        comfy.model_management.load_models_gpu(
            [patcher], memory_required=memory_required,
        )

        H_grid = W_grid = int(predictor.output_resolution)
        num_layers = 2  # SHARP fixed; see model.py:__init__ default

        for b in range(B):
            img_np = image[b].cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image[b])
            if img_np.dtype != np.uint8:
                img_np = (np.clip(img_np, 0, 1) * 255 + 0.5).astype(np.uint8)
            height, width = img_np.shape[:2]
            image_hash = _compute_image_hash(img_np)

            # Cache logic — same as SharpPredict.
            if _encode_cache["image_hash"] == image_hash:
                log.debug(f"  [{b}] cache hit")
                monodepth_output = _monodepth_to(_encode_cache["monodepth_output"], device)
                image_resized_pt = _encode_cache["image_resized"].to(device)
            else:
                log.debug(f"  [{b}] encoding...")
                _encode_cache["image_hash"] = None
                image_pt = (
                    torch.from_numpy(img_np.copy()).float().to(device).permute(2, 0, 1) / 255.0
                )
                image_resized_pt = F.interpolate(
                    image_pt[None],
                    size=(internal_shape[1], internal_shape[0]),
                    mode="bilinear", align_corners=True,
                )
                t0 = time.time()
                monodepth_output, _ = predictor.encode(image_resized_pt)
                log.debug(f"  [{b}] encode time: {time.time() - t0:.2f}s")
                _encode_cache["image_hash"] = image_hash
                _encode_cache["monodepth_output"] = _monodepth_to(monodepth_output, "cpu")
                _encode_cache["image_resized"] = image_resized_pt.cpu()
                _encode_cache["original_shape"] = (height, width)
                comfy.model_management.soft_empty_cache()

            # f_px on the original image, projected to internal_shape's coords.
            # Matches SharpPredict's disparity_factor convention so the decoder
            # produces the same gaussians.
            if intrinsics is not None:
                intr_b = intrinsics[b] if intrinsics.dim() == 3 else intrinsics
                f_px = float(intr_b[0, 0]) * (internal_shape[0] / width)
            else:
                # 30mm 35mm-equiv focal length: f_px = (width / 36) * fl_mm
                f_px = (width / 36.0) * max(0.1, float(focal_length_mm or 30.0))

            disparity_factor = torch.tensor([f_px / width]).float().to(device)
            t1 = time.time()
            gaussians_ndc = predictor.decode(monodepth_output, image_resized_pt, disparity_factor)
            log.debug(f"  [{b}] decode time: {time.time() - t1:.2f}s")

            # Un-flatten to per-pixel grid + split layers.
            l0, l1 = _split_layers_from_flat_gaussians(
                gaussians_ndc, num_layers, H_grid, W_grid,
            )
            # l0, l1 shape: [1, H, W, 14]

            # Depth = position_z = channel index 2.
            d0 = l0[..., 2]  # [1, H, W]
            d1 = l1[..., 2]

            first_depths.append(d0.cpu())
            second_depths.append(d1.cpu())
            first_attrs.append(l0.cpu())
            second_attrs.append(l1.cpu())

        # Stack across batch.
        d0_batch = torch.cat(first_depths, dim=0)  # [B, H, W]
        d1_batch = torch.cat(second_depths, dim=0)
        attrs0_batch = torch.cat(first_attrs, dim=0)  # [B, H, W, 14]
        attrs1_batch = torch.cat(second_attrs, dim=0)

        # IMAGE format: [B, H, W, 3] depth broadcast over channels.
        d0_img = d0_batch.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
        d1_img = d1_batch.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()

        # MULTIBAND_IMAGE: dict with samples [B, 14, H, W] + channel_names.
        l0_mb = _build_multiband(attrs0_batch)
        l1_mb = _build_multiband(attrs1_batch)

        # Rescale intrinsics so their pixel-space matches the emitted depth grid
        # (H_grid x W_grid), not the input face shape. Without this the
        # downstream merger normalizes K by the wrong W and ends up with the
        # wrong effective FOV per face -> visible seams in the merged equirect.
        if intrinsics is not None:
            sx = float(W_grid) / float(width)
            sy = float(H_grid) / float(height)
            intr = intrinsics.detach().clone().float() if isinstance(intrinsics, torch.Tensor) \
                else torch.as_tensor(np.asarray(intrinsics)).float().clone()
            if intr.dim() == 2:
                intr[0, :] *= sx
                intr[1, :] *= sy
                k_dbg = intr
            elif intr.dim() == 3:
                intr[:, 0, :] *= sx
                intr[:, 1, :] *= sy
                k_dbg = intr[0]
            else:
                raise ValueError(f"intrinsics must be (3,3) or (N,3,3), got {tuple(intr.shape)}")
            intrinsics_out = intr
            k_fx_dbg = float(k_dbg[0, 0])
        else:
            intrinsics_out = None
            k_fx_dbg = None

        n_gaussians_total = B * num_layers * H_grid * W_grid
        elapsed = time.time() - t_start
        d0_med = float(d0_batch.median())
        d1_med = float(d1_batch.median())
        k_str = f", K fx→{k_fx_dbg:.1f}" if k_fx_dbg is not None else ""
        log.info(
            f"[SharpPredictGaussianAttrs] {B} face(s) → "
            f"{n_gaussians_total/1e6:.2f}M gaussians "
            f"({B}×{num_layers}×{H_grid}²); "
            f"layer0/1 depth median={d0_med:.2f}m/{d1_med:.2f}m"
            f"{k_str}; {elapsed:.1f}s"
        )

        return io.NodeOutput(
            d0_img, l0_mb, d1_img, l1_mb,
            extrinsics, intrinsics_out,
        )


NODE_CLASS_MAPPINGS = {
    "SharpPredictGaussianAttrs": SharpPredictGaussianAttrs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpPredictGaussianAttrs": "SHARP Predict (Depth + Gaussian Attrs)",
}
