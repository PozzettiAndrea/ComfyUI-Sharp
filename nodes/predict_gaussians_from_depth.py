"""SharpPredictGaussiansFromMetricDepth — image + external depth → PLY paths.

Like SharpPredict, but **directly overrides SHARP's monodepth** with an
externally-provided depth (e.g. from SharpDepthMerge after LSMR-merging
per-face SharpPredictMetricDepth outputs). The learned `depth_alignment`
scale-map UNet is BYPASSED entirely — the external depth feeds straight
into `init_model` as the geometric scaffold, replacing both layer 0
(front) and layer 1 (back) of SHARP's own monodepth.

This is off-label vs the paper's recommended inference (which uses
`depth=None`, no external supervision) — but it's the cleanest way to
inject a multi-view-consistent depth into SHARP's gaussian decoder for
panorama-merging workflows where SHARP's per-face monodepth has scale
ambiguity that the depth_alignment module wasn't trained to handle.

Output convention matches SharpPredict:
  - Single image  → single PLY file at OUTPUT_DIR/{prefix}_{ts}.ply
  - Batch (>1)    → folder OUTPUT_DIR/{prefix}_{ts}/ with NNN.ply files

Wire the batch folder straight into `MergeGaussians (PLY Files)` for a
single merged scene PLY.

Typical pipeline:
  1. SharpPanoramaIcosahedronSplit       → face images
  2. SharpPredictMetricDepth (per face)  → raw per-face metric depths
  3. SharpDepthMerge                     → seam-free equirect depth (LSMR)
  4. SharpPanoramaIcosahedronSplit (on the merged depth IMAGE)
                                          → re-cropped CONSISTENT face depths
  5. SharpPredictGaussiansFromMetricDepth(face_images, consistent_face_depths)
                                          → folder of seam-aligned PLYs
  6. MergeGaussians                      → final unified PLY
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .predict_gaussian_attrs import (
    _compute_image_hash, _monodepth_to, _encode_cache,
)

try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "output",
    )


def _p(msg: str) -> None:
    print(f"[SharpPredictGaussiansFromMetricDepth] {msg}",
          file=sys.stderr, flush=True)


class SharpPredictGaussiansFromMetricDepth(io.ComfyNode):
    """SHARP gaussian decoder driven by external depth → PLY paths."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharpPredictGaussiansFromMetricDepth",
            display_name="SHARP Predict Gaussians (Image + Metric Depth)",
            category="SHARP",
            description=(
                "Run SHARP's gaussian decoder using an externally-provided "
                "metric depth as the alignment target. The external depth "
                "becomes the geometric scaffold for `init_model`'s gaussian "
                "base values, so gaussians share a consistent surface across "
                "multiple views — fixes per-face seams.\n\n"
                "Saves PLYs (single image → one .ply; batch → folder of "
                "NNN.ply). Pipe into MergeGaussians to combine into one PLY."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("SHARP_MODEL_CONFIG").Input("model"),
                io.Image.Input("image"),
                io.Image.Input(
                    "metric_depth",
                    tooltip="External per-face metric depth (any resolution; "
                            "auto-resized to SHARP's internal 1536²). This "
                            "depth REPLACES SHARP's own monodepth for BOTH "
                            "layers (no learned scale-map alignment) — the "
                            "gaussian decoder's `init_model` sees your depth "
                            "directly as the scaffold for layer-0 AND layer-1 "
                            "base values. Layer 1 gaussians end up on the "
                            "same surface as layer 0 (no sky-backplate "
                            "hallucination), the per-layer prediction head's "
                            "deltas can still drift them sub-pixel in xy + "
                            "color so layer 1 serves as a view-dependent "
                            "appearance variation. Typically: "
                            "SharpPredictMetricDepth → SharpDepthMerge → "
                            "SharpPanoramaIcosahedronSplit (depth as IMAGE)."),
                io.Float.Input(
                    "focal_length_mm", default=30.0, min=0.0, max=500.0,
                    step=0.1, optional=True,
                    tooltip="Focal length in mm (35mm equiv). Ignored if "
                            "intrinsics provided."),
                io.String.Input(
                    "output_prefix", default="sharp_aligned", optional=True,
                    tooltip="Prefix for PLY filename or batch folder name."),
                io.Boolean.Input(
                    "save_background_layer", default=True, optional=True,
                    tooltip=(
                        "SHARP emits 2 gaussian layers per pixel: layer 0 "
                        "(visible/front surface) and layer 1 (back/occluded). "
                        "True (default): save both layers — 2× the gaussian "
                        "count, captures hidden surfaces revealed by parallax. "
                        "False: save only layer 0 — halves the gaussian count "
                        "(768² per face instead of 768²×2). Use False when "
                        "you'll voxel-dedup downstream anyway, or when the "
                        "scene has minimal occlusion (open outdoor)."
                    )),
                io.Combo.Input(
                    "snap_to_external_depth",
                    options=["z_only", "xyz_full", "none"],
                    default="z_only",
                    tooltip=(
                        "Force gaussian positions to exactly match the external "
                        "depth in NDC space (applied after gaussian_composer, "
                        "before unprojection).\n\n"
                        "  z_only (recommended): scale each gaussian by "
                        "target_z/current_z. The NDC xy ray is preserved (so "
                        "the prediction head's sub-pixel xy drift survives — "
                        "layer 1 still varies in appearance from layer 0), but "
                        "depth is forced to match external.\n\n"
                        "  xyz_full: reset xy to the pixel-grid identity AND "
                        "set depth to external. Every gaussian sits exactly "
                        "on the back-projected ray of its source pixel at the "
                        "external depth — no learned drift at all. Most rigid; "
                        "layer 0 and layer 1 collapse to identical positions.\n\n"
                        "  none: skip the snap entirely — trust the gaussian "
                        "decoder's predicted z (the direct-monodepth-override "
                        "behavior is in effect but not enforced)."
                    )),
                io.Custom("EXTRINSICS").Input(
                    "extrinsics", optional=True,
                    tooltip="Per-face extrinsics from "
                            "SharpPanoramaIcosahedronSplit. Must match "
                            "image batch size if batched."),
                io.Custom("INTRINSICS").Input(
                    "intrinsics", optional=True,
                    tooltip="Per-face intrinsics from "
                            "SharpPanoramaIcosahedronSplit (pixel-K for the "
                            "ORIGINAL face image resolution, not the merged "
                            "depth resolution). If absent, focal_length_mm "
                            "is used."),
                io.Float.Input(
                    "edge_crop", default=0.0, min=0.0, max=0.5, step=0.01,
                    optional=True,
                    tooltip="Fraction of each edge to drop before saving the "
                            "PLY (per side). 0.0 = no crop (default), 0.1 = "
                            "drop the outer 10%% of width AND height (keep the "
                            "central 80%% × 80%% gaussians). Useful to discard "
                            "edge gaussians that SHARP often hallucinates "
                            "at silhouettes / frame edges, especially when "
                            "stitching panorama faces — adjacent faces "
                            "overlap, so cropping a few percent off each "
                            "removes the worst-quality region without losing "
                            "scene coverage. Applied per-layer to the H×W "
                            "pixel grid so both layer 0 and layer 1 keep "
                            "the same central region."),
            ],
            outputs=[
                io.String.Output(
                    display_name="ply_path",
                    tooltip="Single image: path to {prefix}_{ts}.ply. "
                            "Batch: path to the folder (identical to "
                            "ply_folder output)."),
                io.String.Output(
                    display_name="ply_folder",
                    tooltip="Always the containing folder of the PLY(s): "
                            "OUTPUT_DIR/{prefix}_{ts}/ for batch, or the "
                            "parent dir for single-image runs. Wire into "
                            "MergeGaussians.ply_folder regardless of batch size."),
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
        metric_depth: torch.Tensor,
        focal_length_mm: float = 30.0,
        output_prefix: str = "sharp_aligned",
        save_background_layer: bool = True,
        snap_to_external_depth: str = "z_only",
        edge_crop: float = 0.0,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ):
        global _encode_cache
        import comfy.model_management
        import comfy.utils
        from .load_model import _load_sharp_model
        from .sharp.gaussians import save_ply, unproject_gaussians

        patcher = _load_sharp_model(model)
        predictor = patcher.model
        device = patcher.load_device

        if image.dim() == 3:
            image = image.unsqueeze(0)
        B = image.shape[0]

        if metric_depth.dim() == 3:
            metric_depth = metric_depth.unsqueeze(0)
        if metric_depth.shape[0] != B:
            raise ValueError(
                f"metric_depth batch {metric_depth.shape[0]} != image batch {B}"
            )

        # Sanity-check the alignment is wired through the model.
        scale_map_estimator = predictor.depth_alignment.scale_map_estimator
        if scale_map_estimator is None:
            _p("WARN: scale_map_estimator not in model; depth_alignment may "
               "no-op and gaussians may not benefit from the external depth.")

        internal_shape = (1536, 1536)
        H_grid = W_grid = int(predictor.output_resolution)  # typ. 768
        input_shape = [1, 3, internal_shape[0], internal_shape[1]]
        memory_required = patcher.memory_required(input_shape)
        comfy.model_management.load_models_gpu(
            [patcher], memory_required=memory_required,
        )

        # Output path(s) — same convention as SharpPredict.
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        if B == 1:
            output_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_{timestamp}.ply")
            output_folder = None
            is_batch = False
        else:
            output_folder = os.path.join(OUTPUT_DIR, f"{output_prefix}_{timestamp}")
            os.makedirs(output_folder, exist_ok=True)
            output_path = output_folder
            is_batch = True

        all_ply_paths = []
        all_extrinsics = []
        all_intrinsics = []

        pbar = comfy.utils.ProgressBar(B)
        t_start = time.time()
        n_gaussians_total = 0

        for b in range(B):
            comfy.model_management.throw_exception_if_processing_interrupted()

            img_np = image[b].cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image[b])
            if img_np.dtype != np.uint8:
                img_np = (np.clip(img_np, 0, 1) * 255 + 0.5).astype(np.uint8)
            height, width = img_np.shape[:2]
            image_hash = _compute_image_hash(img_np)

            # Encode (shared cache).
            if _encode_cache["image_hash"] == image_hash:
                monodepth_output = _monodepth_to(_encode_cache["monodepth_output"], device)
                image_resized_pt = _encode_cache["image_resized"].to(device)
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

            # Focal-length resolution. intrinsics is pixel-K for `width` (the
            # face image resolution). disparity_factor matches SharpPredict.
            if intrinsics is not None:
                intr_b = intrinsics[b] if intrinsics.dim() == 3 else intrinsics
                f_px = float(intr_b[0, 0]) * (internal_shape[0] / width)
            else:
                # Match SharpPredict's `convert_focallength` formula so the
                # gaussian outputs are consistent across all Sharp predict
                # nodes (was previously off by ~1.18× on square inputs).
                from .utils.image import convert_focallength as _cfl
                f_px = float(_cfl(width, height, max(0.1, float(focal_length_mm or 30.0))))
            disparity_factor = torch.tensor([f_px / width]).float().to(device)

            # External depth → 1536², broadcast to BOTH layers.
            md_b = metric_depth[b]
            if md_b.dim() == 3:
                md_b = md_b[..., 0]
            md_b = md_b.to(device).float().unsqueeze(0).unsqueeze(0)
            external_depth = F.interpolate(
                md_b, size=internal_shape, mode="bilinear", align_corners=True,
            )
            # Both monodepth layers = external depth → no back-surface
            # hallucination. Layer 1's gaussians collapse onto the same
            # surface as layer 0; the prediction head's per-layer deltas
            # can still drift them sub-pixel in xy and color, so layer 1
            # serves as the view-dependent appearance variation rather
            # than an occlusion backplate (consistent with how SHARP uses
            # layer 1 as an SH-coefficient substitute per the paper).
            monodepth_override = external_depth.repeat(1, 2, 1, 1)  # [1, 2, 1536, 1536]

            # Direct override: bypass `depth_alignment` (the learned scale-
            # map UNet). The paper's recommended inference path passes
            # depth=None, leaving alignment as identity and using SHARP's
            # own monodepth. Here we instead replace monodepth entirely
            # with the user's external depth — same code path init_model →
            # feature_model → prediction_head → gaussian_composer that the
            # official `decode()` runs, just with external monodepth
            # instead of `disparity_factor / disparity`.
            init_output = predictor.init_model(image_resized_pt, monodepth_override)
            image_features = predictor.feature_model(
                init_output.feature_input,
                encodings=monodepth_output.output_features,
            )
            delta_values = predictor.prediction_head(image_features)
            gaussians_ndc = predictor.gaussian_composer(
                delta=delta_values,
                base_values=init_output.gaussian_base_values,
                global_scale=init_output.global_scale,
            )

            # ----- Hard-snap gaussian positions to external depth (NDC). -----
            # Compute target NDC state (target_z from external depth, target
            # xy from the pixel-grid identity) for both stats logging and the
            # actual snap. Composer flatten order is (layer, h, w).
            N_total = int(gaussians_ndc.mean_vectors.shape[1])
            num_layers = N_total // (H_grid * W_grid)

            ext_at_grid = F.adaptive_avg_pool2d(
                external_depth, output_size=(H_grid, W_grid),
            )[0, 0]  # [H_grid, W_grid]
            target_z_flat = (
                ext_at_grid.unsqueeze(0)
                .expand(num_layers, H_grid, W_grid)
                .reshape(-1)
            )  # [N_total]

            stride = internal_shape[0] // H_grid
            xs_ndc = torch.arange(
                0.5 * stride, internal_shape[1], stride,
                device=device, dtype=torch.float32,
            )
            ys_ndc = torch.arange(
                0.5 * stride, internal_shape[0], stride,
                device=device, dtype=torch.float32,
            )
            xs_ndc = 2.0 * xs_ndc / internal_shape[1] - 1.0
            ys_ndc = 2.0 * ys_ndc / internal_shape[0] - 1.0
            base_xx, base_yy = torch.meshgrid(xs_ndc, ys_ndc, indexing="xy")
            base_xx_flat = (
                base_xx.unsqueeze(0)
                .expand(num_layers, H_grid, W_grid)
                .reshape(-1)
            )
            base_yy_flat = (
                base_yy.unsqueeze(0)
                .expand(num_layers, H_grid, W_grid)
                .reshape(-1)
            )

            # Current NDC state: mean_vectors = (xx_ndc*z, yy_ndc*z, z).
            mv = gaussians_ndc.mean_vectors[0]  # [N_total, 3]
            current_z = mv[:, 2].clamp(min=1e-6)
            current_xx_ndc = mv[:, 0] / current_z
            current_yy_ndc = mv[:, 1] / current_z

            # Stats: report would-be shifts at 10 random gaussians + aggregate.
            # Print once per face (we have up to 42 faces — full per-face dump
            # would be noisy).
            sample_g = torch.Generator(device="cpu").manual_seed(int(b))
            sample_idx = torch.randperm(N_total, generator=sample_g)[:10]
            _p(f"--- snap stats face {b} (mode={snap_to_external_depth}, "
               f"would-be shifts at 10 random gaussians) ---")
            _p(f"{'idx':>9} | {'lyr':>3} {'h':>4} {'w':>4} | "
               f"{'z_now':>8} {'z_tgt':>8} {'dz':>8} | "
               f"{'x_now':>7} {'x_tgt':>7} {'dx':>7} | "
               f"{'y_now':>7} {'y_tgt':>7} {'dy':>7}")
            for idx_t in sample_idx.tolist():
                lyr = idx_t // (H_grid * W_grid)
                rem = idx_t % (H_grid * W_grid)
                hh = rem // W_grid
                ww = rem % W_grid
                z_now = float(current_z[idx_t])
                z_tgt = float(target_z_flat[idx_t])
                x_now = float(current_xx_ndc[idx_t])
                x_tgt = float(base_xx_flat[idx_t])
                y_now = float(current_yy_ndc[idx_t])
                y_tgt = float(base_yy_flat[idx_t])
                _p(f"{idx_t:>9d} | {lyr:>3d} {hh:>4d} {ww:>4d} | "
                   f"{z_now:>8.3f} {z_tgt:>8.3f} {z_tgt - z_now:>+8.3f} | "
                   f"{x_now:>+7.4f} {x_tgt:>+7.4f} {x_tgt - x_now:>+7.4f} | "
                   f"{y_now:>+7.4f} {y_tgt:>+7.4f} {y_tgt - y_now:>+7.4f}")
            dz_all = (target_z_flat - current_z).abs()
            dx_all = (base_xx_flat - current_xx_ndc).abs()
            dy_all = (base_yy_flat - current_yy_ndc).abs()
            _p(f"aggregate |shift| over all {N_total} gaussians: "
               f"|dz| mean={float(dz_all.mean()):.4f} "
               f"max={float(dz_all.max()):.4f} | "
               f"|dx_ndc| mean={float(dx_all.mean()):.5f} "
               f"max={float(dx_all.max()):.5f} | "
               f"|dy_ndc| mean={float(dy_all.mean()):.5f} "
               f"max={float(dy_all.max()):.5f}")

            if snap_to_external_depth == "z_only":
                # Proportional rescale: mean_vectors *= target_z/current_z.
                # (xx*z, yy*z, z) → (xx*target_z, yy*target_z, target_z).
                scale = (target_z_flat / current_z).unsqueeze(0).unsqueeze(-1)
                new_mean_vectors = gaussians_ndc.mean_vectors * scale
                gaussians_ndc = gaussians_ndc._replace(
                    mean_vectors=new_mean_vectors,
                )
            elif snap_to_external_depth == "xyz_full":
                # Reset xy to pixel-grid identity AND set z to target.
                new_mean_vectors = torch.stack(
                    [
                        base_xx_flat * target_z_flat,
                        base_yy_flat * target_z_flat,
                        target_z_flat,
                    ],
                    dim=-1,
                ).unsqueeze(0)  # [1, N_total, 3]
                gaussians_ndc = gaussians_ndc._replace(
                    mean_vectors=new_mean_vectors,
                )
            # else: "none" → leave gaussians_ndc as-is.

            # Build extrinsics + intrinsics_resized for unprojection.
            if extrinsics is not None:
                ext_b = extrinsics[b] if extrinsics.dim() == 3 else extrinsics
                unproj_extrinsics = ext_b.to(device).float()
            else:
                unproj_extrinsics = torch.eye(4, device=device)

            if intrinsics is not None:
                intr_b = intrinsics[b] if intrinsics.dim() == 3 else intrinsics
                K = intr_b.to(device).float().clone()
                # Promote to 4x4 if it came as 3x3 (unproject_gaussians wants 4x4).
                if K.shape == (3, 3):
                    K4 = torch.eye(4, device=device, dtype=K.dtype)
                    K4[:3, :3] = K
                    K = K4
            else:
                K = torch.tensor(
                    [
                        [f_px, 0,    width / 2,  0],
                        [0,    f_px, height / 2, 0],
                        [0,    0,    1,          0],
                        [0,    0,    0,          1],
                    ],
                    dtype=torch.float32, device=device,
                )
            intrinsics_resized = K.clone()
            intrinsics_resized[0] *= internal_shape[0] / width
            intrinsics_resized[1] *= internal_shape[1] / height

            gaussians = unproject_gaussians(
                gaussians_ndc, unproj_extrinsics, intrinsics_resized, internal_shape,
            )

            # Optionally drop layer 1 (back/occluded surface).
            # The gaussian_composer's flatten order is (layer, h, w), so the
            # first H_grid*W_grid entries are layer 0, the second are layer 1.
            n_layer0 = H_grid * W_grid
            if not save_background_layer:
                gaussians = gaussians._replace(
                    mean_vectors=gaussians.mean_vectors[:, :n_layer0],
                    singular_values=gaussians.singular_values[:, :n_layer0],
                    quaternions=gaussians.quaternions[:, :n_layer0],
                    colors=gaussians.colors[:, :n_layer0],
                    opacities=gaussians.opacities[:, :n_layer0],
                )

            # Optional edge crop: drop gaussians within `edge_crop` of any
            # edge of the H_grid × W_grid pixel grid. Applied per-layer so
            # both layers keep the same central region. Cheap (single
            # boolean mask + tensor subset) and physically removes the
            # cropped gaussians from the PLY (smaller file, faster render).
            if edge_crop and edge_crop > 0.0:
                crop_h = int(round(float(edge_crop) * H_grid))
                crop_w = int(round(float(edge_crop) * W_grid))
                if crop_h * 2 >= H_grid or crop_w * 2 >= W_grid:
                    raise ValueError(
                        f"edge_crop={edge_crop} crops entire grid "
                        f"(crop_h={crop_h}, crop_w={crop_w}, "
                        f"grid={H_grid}×{W_grid})"
                    )
                mask_2d = torch.zeros(
                    H_grid, W_grid, dtype=torch.bool,
                    device=gaussians.mean_vectors.device,
                )
                mask_2d[crop_h:H_grid - crop_h, crop_w:W_grid - crop_w] = True
                mask_per_layer = mask_2d.flatten()              # [H*W]
                # If layer 1 was kept, repeat the mask for the second layer.
                if save_background_layer:
                    keep_mask = torch.cat([mask_per_layer, mask_per_layer], dim=0)
                else:
                    keep_mask = mask_per_layer
                n_before = int(gaussians.mean_vectors.shape[1])
                gaussians = gaussians._replace(
                    mean_vectors=gaussians.mean_vectors[:, keep_mask],
                    singular_values=gaussians.singular_values[:, keep_mask],
                    quaternions=gaussians.quaternions[:, keep_mask],
                    colors=gaussians.colors[:, keep_mask],
                    opacities=gaussians.opacities[:, keep_mask],
                )
                n_after = int(gaussians.mean_vectors.shape[1])
                if b == 0:  # log once per batch to avoid spam on 42 faces
                    _p(
                        f"edge_crop={edge_crop:.3f} → crop_h={crop_h} crop_w={crop_w} "
                        f"per side; kept {n_after}/{n_before} gaussians "
                        f"({100.0 * n_after / max(n_before, 1):.1f}%) per face"
                    )

            # Save PLY.
            if is_batch:
                ply_path = os.path.join(output_folder, f"{b+1:03d}.ply")
            else:
                ply_path = output_path
            _, metadata = save_ply(gaussians, f_px, (height, width), Path(ply_path))

            all_ply_paths.append(ply_path)
            all_extrinsics.append(metadata["extrinsic"])
            all_intrinsics.append(metadata["intrinsic"])
            n_gaussians_total += int(metadata["num_gaussians"])
            pbar.update(1)

        elapsed = time.time() - t_start
        loc = output_folder if is_batch else output_path
        layer_str = "layer0 only" if not save_background_layer else "both layers"
        if edge_crop and edge_crop > 0.0:
            layer_str += f", edge_crop={edge_crop:.3f}"
        _p(
            f"{B} face(s) → {n_gaussians_total/1e6:.2f}M gaussians total "
            f"({layer_str}, snap={snap_to_external_depth}); "
            f"saved to {loc}; {elapsed:.1f}s"
        )

        # ply_folder is always the directory holding the PLYs (regardless
        # of single/batch), so downstream MergeGaussians can wire to it
        # without caring about batch size.
        ply_folder_out = output_folder if is_batch else os.path.dirname(output_path)

        return io.NodeOutput(
            output_path, ply_folder_out, all_extrinsics[0], all_intrinsics[0],
        )


NODE_CLASS_MAPPINGS = {
    "SharpPredictGaussiansFromMetricDepth": SharpPredictGaussiansFromMetricDepth,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpPredictGaussiansFromMetricDepth": "SHARP Predict Gaussians (Image + Metric Depth)",
}
