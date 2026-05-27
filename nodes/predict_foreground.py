"""SharpPredictForeground node for ComfyUI-Sharp.

Same as SharpPredict, but emits ONLY layer 0 (visible/front surface) — drops
layer 1 (back/occluded surface). Halves the per-image gaussian count
(768²×1 instead of 768²×2). Use when you don't need backside detail and want
smaller PLYs / faster downstream rendering.

The gaussian_composer's flatten order is (layer, h, w), so the first
H_grid × W_grid entries of the returned Gaussians3D are layer 0; we slice
to that range before save_ply.
"""

import logging
import os
import time
from pathlib import Path

import torch

from comfy_api.latest import io

from .predict import _predict_image_cached
from .utils.image import comfy_to_numpy_rgb, convert_focallength

log = logging.getLogger("sharp")

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
            log.info(
                f"Processing {batch_size} image(s) with provided camera "
                f"parameters (foreground-only, layer 0 = {n_layer0} gaussians/face)"
            )
        else:
            log.info(
                f"Processing {batch_size} image(s) (foreground-only, "
                f"layer 0 = {n_layer0} gaussians/face)"
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
            gaussians = _predict_image_cached(
                patcher, predictor, image_np, f_px, device,
                extrinsics=img_extrinsics,
                intrinsics=img_intrinsics,
            )

            # Foreground-only slice: drop layer 1 (occluded surface).
            # Composer flatten order is (layer, h, w), so [0:n_layer0] = layer 0.
            gaussians = gaussians._replace(
                mean_vectors=gaussians.mean_vectors[:, :n_layer0],
                singular_values=gaussians.singular_values[:, :n_layer0],
                quaternions=gaussians.quaternions[:, :n_layer0],
                colors=gaussians.colors[:, :n_layer0],
                opacities=gaussians.opacities[:, :n_layer0],
            )

            if is_batch:
                ply_filename = f"{i+1:03d}.ply"
                ply_path = os.path.join(output_folder, ply_filename)
            else:
                ply_path = output_path

            _, metadata = save_ply(gaussians, f_px, (height, width), Path(ply_path))

            all_ply_paths.append(ply_path)
            all_extrinsics.append(metadata["extrinsic"])
            all_intrinsics.append(metadata["intrinsic"])

            log.info(f"Saved: {ply_path} ({metadata['num_gaussians']:,} gaussians, layer 0)")
            pbar.update(1)

        inference_time = time.time() - inference_start
        log.info(
            f"Total inference time: {inference_time:.2f}s "
            f"({inference_time/batch_size:.2f}s per image)"
        )

        return io.NodeOutput(output_path, all_extrinsics[0], all_intrinsics[0])


NODE_CLASS_MAPPINGS = {
    "SharpPredictForeground": SharpPredictForeground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpPredictForeground": "SHARP Predict Foreground (Image to PLY)",
}
