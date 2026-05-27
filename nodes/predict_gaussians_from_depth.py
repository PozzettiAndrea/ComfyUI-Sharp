"""SharpPredictGaussiansFromMetricDepth — image + external depth → PLY paths.

Like SharpPredict, but feeds an externally-provided metric depth into
`predictor.decode(..., depth=...)`. The decoder's depth_alignment module
(`model.py:2023`) blends the raw monodepth with the external depth before
init_model initializes gaussian base values, so the resulting gaussians
share a consistent geometric scaffold across multiple views — fixes the
per-face seams caused by per-view monocular scale ambiguity.

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
                            "auto-resized to SHARP's internal 1536² before "
                            "alignment). Typically: SharpPredictMetricDepth "
                            "→ SharpDepthMerge → SharpPanoramaIcosahedronSplit "
                            "(depth as IMAGE)."),
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
                f_px = (width / 36.0) * max(0.1, float(focal_length_mm or 30.0))
            disparity_factor = torch.tensor([f_px / width]).float().to(device)

            # External depth → 1536² (alignment target).
            md_b = metric_depth[b]
            if md_b.dim() == 3:
                md_b = md_b[..., 0]
            md_b = md_b.to(device).float().unsqueeze(0).unsqueeze(0)
            external_depth = F.interpolate(
                md_b, size=internal_shape, mode="bilinear", align_corners=True,
            )

            # Decode WITH depth alignment.
            gaussians_ndc = predictor.decode(
                monodepth_output, image_resized_pt, disparity_factor,
                depth=external_depth,
            )

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
            if not save_background_layer:
                n_layer0 = H_grid * W_grid
                gaussians = gaussians._replace(
                    mean_vectors=gaussians.mean_vectors[:, :n_layer0],
                    singular_values=gaussians.singular_values[:, :n_layer0],
                    quaternions=gaussians.quaternions[:, :n_layer0],
                    colors=gaussians.colors[:, :n_layer0],
                    opacities=gaussians.opacities[:, :n_layer0],
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
        _p(
            f"{B} face(s) → {n_gaussians_total/1e6:.2f}M gaussians total "
            f"({layer_str}, depth-aligned); saved to {loc}; {elapsed:.1f}s"
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
