"""SharpPredictForeground node for ComfyUI-Sharp.

Same as SharpPredict, but emits a single gaussian per pixel instead of
SHARP's native 2 (halves PLY size, makes "foreground only" use cases
practical). Two selection modes:

- **closest** (default): per pixel, keep the gaussian with the smaller
  world-space z (closer to camera). SHARP's monodepth output IS sorted
  near->far at the head (`model.py:1122-1125`: `disparity.max` for layer
  0, `disparity.min` for layer 1), but the gaussian decoder adds an
  unconstrained delta_z per layer — so layer 0 isn't reliably the front
  post-decode. `closest` looks at the actual z and picks per-pixel,
  guaranteeing the kept gaussian is the front surface.

- **slice**: trust the layer index — keep the first H·W gaussians
  (layer 0) and drop the rest. Faster (no per-pixel comparison) and
  matches what the file used to do; correct ~for most pixels but the
  gaussian decoder can swap layers (especially at sky / edge / uncertain
  regions), so a fraction of "foreground" pixels are silently dropped.

In both modes the output is the front gaussian per pixel — only the
selection rule differs.
"""

import logging
import os
import sys
import time
from pathlib import Path

import torch

from comfy_api.latest import io

from .predict import _predict_image_cached
from .utils.image import comfy_to_numpy_rgb, convert_focallength

log = logging.getLogger("sharp")


def _p(msg: str) -> None:
    """Print to stderr so the line surfaces in ComfyUI's worker log
    (`log.info` on the 'sharp' logger doesn't always route through the
    comfy-env subprocess pipe — direct print to stderr does)."""
    print(f"[SharpPredictForeground] {msg}", file=sys.stderr, flush=True)

try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


class SharpPredictForeground(io.ComfyNode):
    """Run SHARP inference and save PLY containing ONLY layer-0 gaussians.

    Identical wiring to SharpPredict (same inputs, same outputs). The only
    difference is the foreground-only slice applied before save_ply.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharpPredictForeground",
            display_name="SHARP Predict Foreground (Image to PLY)",
            category="SHARP",
            description=(
                "Generate 3D Gaussian Splatting PLY file(s) from image(s) "
                "using SHARP, keeping only layer-0 (visible/front surface) "
                "gaussians and dropping layer-1 (back/occluded). Halves the "
                "per-image gaussian count vs SharpPredict. Batch input creates "
                "a folder with numbered PLY files."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("SHARP_MODEL_CONFIG").Input("model"),
                io.Image.Input("image"),
                io.Combo.Input(
                    "mode", options=["closest", "slice"], default="closest",
                    tooltip=(
                        "How to reduce SHARP's 2-layer gaussian output to 1 "
                        "per pixel.\n"
                        "  closest: pick the gaussian with smaller z per "
                        "pixel (guaranteed front surface — robust to "
                        "decoder layer swaps).\n"
                        "  slice: keep layer 0 only (first H·W gaussians); "
                        "faster but misses pixels where the gaussian "
                        "decoder swapped layers."
                    )),
                io.Float.Input(
                    "focal_length_mm", default=30.0, min=0.0, max=500.0,
                    step=0.1, optional=True,
                    tooltip="Focal length in mm (35mm equiv). 0 = auto (30mm). "
                            "Ignored if intrinsics provided."),
                io.String.Input(
                    "output_prefix", default="sharp_fg", optional=True,
                    tooltip="Prefix for output PLY filename or batch folder."),
                io.Custom("EXTRINSICS").Input(
                    "extrinsics", optional=True,
                    tooltip="Camera extrinsics (from SamplePanorama). If "
                            "batched, must match image batch size."),
                io.Custom("INTRINSICS").Input(
                    "intrinsics", optional=True,
                    tooltip="Camera intrinsics. Overrides focal_length_mm."),
            ],
            outputs=[
                io.String.Output(display_name="ply_path"),
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
        mode: str = "closest",
        focal_length_mm: float = 0.0,
        output_prefix: str = "sharp_fg",
        extrinsics: torch.Tensor = None,
        intrinsics: torch.Tensor = None,
    ):
        import comfy.model_management
        import comfy.utils
        from .load_model import _load_sharp_model
        from .sharp.gaussians import save_ply

        patcher = _load_sharp_model(model)
        predictor = patcher.model
        device = patcher.load_device

        if image.dim() == 3:
            image = image.unsqueeze(0)
        batch_size = image.shape[0]

        # Layer-0 count = output_resolution².
        H_grid = W_grid = int(predictor.output_resolution)
        n_layer0 = H_grid * W_grid

        has_camera_params = extrinsics is not None and intrinsics is not None
        if has_camera_params:
            if extrinsics.dim() == 2:
                extrinsics = extrinsics.unsqueeze(0)
            if extrinsics.shape[0] != batch_size:
                raise ValueError(
                    f"Extrinsics batch size ({extrinsics.shape[0]}) must match "
                    f"image batch size ({batch_size})"
                )
            _p(
                f"processing {batch_size} image(s) with provided camera params "
                f"(mode={mode}, {n_layer0} gaussians/face)"
            )
        else:
            _p(
                f"processing {batch_size} image(s) (mode={mode}, "
                f"{n_layer0} gaussians/face)"
            )

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)

        if batch_size == 1:
            output_filename = f"{output_prefix}_{timestamp}.ply"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            output_folder = None
            is_batch = False
        else:
            folder_name = f"{output_prefix}_{timestamp}"
            output_folder = os.path.join(OUTPUT_DIR, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            output_path = output_folder
            is_batch = True

        all_ply_paths = []
        all_extrinsics = []
        all_intrinsics = []

        inference_start = time.time()
        pbar = comfy.utils.ProgressBar(batch_size)

        for i in range(batch_size):
            comfy.model_management.throw_exception_if_processing_interrupted()
            single_image = image[i:i+1]
            image_np = comfy_to_numpy_rgb(single_image)
            height, width = image_np.shape[:2]

            if i == 0:
                log.info(f"Image size: {width}x{height}")

            if has_camera_params:
                img_intrinsics = intrinsics.to(device)
                img_extrinsics = extrinsics[i].to(device)
                f_px = img_intrinsics[0, 0].item()
            else:
                if focal_length_mm > 0:
                    f_px = convert_focallength(width, height, focal_length_mm)
                else:
                    f_px = convert_focallength(width, height, 30.0)
                img_extrinsics = None
                img_intrinsics = None

            log.info(f"Running inference on image {i+1}/{batch_size}...")
            gaussians, _refined_l0, _refined_l1 = _predict_image_cached(
                patcher, predictor, image_np, f_px, device,
                extrinsics=img_extrinsics,
                intrinsics=img_intrinsics,
            )
            # foreground-only path ignores both depth side-outputs

            # Reduce SHARP's 2-layer output to a single gaussian per pixel.
            # Composer flatten order is (layer, h, w): indices [0:n_layer0]
            # are layer 0, [n_layer0:2*n_layer0] are layer 1, both aligned
            # to the same pixel grid (gaussian i of layer 0 corresponds to
            # gaussian n_layer0+i of layer 1 at the same (h, w)).
            if mode == "slice":
                gaussians = gaussians._replace(
                    mean_vectors=gaussians.mean_vectors[:, :n_layer0],
                    singular_values=gaussians.singular_values[:, :n_layer0],
                    quaternions=gaussians.quaternions[:, :n_layer0],
                    colors=gaussians.colors[:, :n_layer0],
                    opacities=gaussians.opacities[:, :n_layer0],
                )
            elif mode == "closest":
                # Per pixel: pick the layer whose gaussian has smaller z
                # (= closer to camera in world coords after unprojection).
                z0 = gaussians.mean_vectors[:, :n_layer0, 2]               # [B, n_layer0]
                z1 = gaussians.mean_vectors[:, n_layer0:2*n_layer0, 2]     # [B, n_layer0]
                pick_l0 = z0 <= z1                                          # [B, n_layer0]
                layer0_idx = torch.arange(n_layer0, device=z0.device)
                layer1_idx = layer0_idx + n_layer0
                pick_idx = torch.where(pick_l0, layer0_idx[None], layer1_idx[None])  # [B, n_layer0]

                def _gather2d(t):  # t: [B, 2*n_layer0, C]
                    idx = pick_idx.unsqueeze(-1).expand(-1, -1, t.shape[-1])
                    return torch.gather(t, 1, idx)
                gaussians = gaussians._replace(
                    mean_vectors=_gather2d(gaussians.mean_vectors),
                    singular_values=_gather2d(gaussians.singular_values),
                    quaternions=_gather2d(gaussians.quaternions),
                    colors=_gather2d(gaussians.colors),
                    opacities=torch.gather(gaussians.opacities, 1, pick_idx),
                )
                _swap_pct = 100.0 * (1.0 - pick_l0.float().mean().item())
                _p(
                    f"image {i+1}/{batch_size} [closest]: kept layer 0 at "
                    f"{100.0 - _swap_pct:.1f}% of pixels, swapped to layer 1 "
                    f"at {_swap_pct:.1f}% (decoder-swapped pixels)"
                )
            else:
                raise ValueError(f"unknown mode {mode!r}; expected 'closest' or 'slice'")

            if is_batch:
                ply_filename = f"{i+1:03d}.ply"
                ply_path = os.path.join(output_folder, ply_filename)
            else:
                ply_path = output_path

            _, metadata = save_ply(gaussians, f_px, (height, width), Path(ply_path))

            all_ply_paths.append(ply_path)
            all_extrinsics.append(metadata["extrinsic"])
            all_intrinsics.append(metadata["intrinsic"])

            _p(f"saved {ply_path} ({metadata['num_gaussians']:,} gaussians, mode={mode})")
            pbar.update(1)

        inference_time = time.time() - inference_start
        _p(
            f"done. total inference {inference_time:.2f}s "
            f"({inference_time/batch_size:.2f}s per image)"
        )

        return io.NodeOutput(output_path, all_extrinsics[0], all_intrinsics[0])


NODE_CLASS_MAPPINGS = {
    "SharpPredictForeground": SharpPredictForeground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpPredictForeground": "SHARP Predict Foreground (Image to PLY)",
}
