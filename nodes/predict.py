"""SharpPredict node for ComfyUI-Sharp."""

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Try to import ComfyUI folder_paths for output directory
try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

from ..utils.image import comfy_to_numpy_rgb, convert_focallength


class SharpPredict:
    """Run SHARP inference to generate 3D Gaussians from a single image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SHARP_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "focal_length_mm": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 500.0,
                    "step": 0.1,
                    "tooltip": "Focal length in mm (35mm equivalent). 0 = auto (defaults to 30mm)."
                }),
                "output_prefix": ("STRING", {
                    "default": "sharp",
                    "tooltip": "Prefix for output PLY filename."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "EXTRINSICS", "INTRINSICS",)
    RETURN_NAMES = ("ply_path", "extrinsics", "intrinsics",)
    FUNCTION = "predict"
    CATEGORY = "SHARP"
    OUTPUT_NODE = True
    DESCRIPTION = "Generate 3D Gaussian Splatting PLY file from a single image using SHARP."

    @torch.no_grad()
    def predict(
        self,
        model: dict,
        image: torch.Tensor,
        focal_length_mm: float = 0.0,
        output_prefix: str = "sharp",
    ):
        """Run SHARP inference and save PLY file."""
        from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians

        predictor = model["predictor"]
        device = torch.device(model["device"])

        # Convert ComfyUI image to numpy RGB
        image_np = comfy_to_numpy_rgb(image)
        height, width = image_np.shape[:2]
        print(f"[SHARP] Input image: {width}x{height}")

        # Determine focal length in pixels
        if focal_length_mm > 0:
            f_px = convert_focallength(width, height, focal_length_mm)
        else:
            # Default to 30mm equivalent
            f_px = convert_focallength(width, height, 30.0)

        # Run inference
        print(f"[SHARP] Running inference...")
        inference_start = time.time()
        gaussians = self._predict_image(predictor, image_np, f_px, device)
        inference_time = time.time() - inference_start
        print(f"[SHARP] Inference complete in {inference_time:.2f}s")

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Generate output filename
        timestamp = int(time.time() * 1000)
        output_filename = f"{output_prefix}_{timestamp}.ply"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Save PLY and get metadata
        print(f"[SHARP] Saving PLY file...")
        _, metadata = save_ply(gaussians, f_px, (height, width), Path(output_path))

        print(f"[SHARP] Saved: {output_path}")
        print(f"[SHARP] Result: {metadata['num_gaussians']:,} gaussians")

        # Extract extrinsics (4x4) and intrinsics (3x3) for camera setup
        extrinsics = metadata["extrinsic"]  # 4x4 identity matrix
        intrinsics = metadata["intrinsic"]  # 3x3 camera matrix

        return (output_path, extrinsics, intrinsics,)

    def _predict_image(
        self,
        predictor,
        image: np.ndarray,
        f_px: float,
        device: torch.device,
    ):
        """Predict Gaussians from an image.

        Based on sharp/cli/predict.py:predict_image()
        """
        from sharp.utils.gaussians import unproject_gaussians

        internal_shape = (1536, 1536)

        # Convert to tensor and normalize
        image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
        _, height, width = image_pt.shape
        disparity_factor = torch.tensor([f_px / width]).float().to(device)

        # Resize to internal resolution
        image_resized_pt = F.interpolate(
            image_pt[None],
            size=(internal_shape[1], internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )

        # Predict Gaussians in NDC space
        gaussians_ndc = predictor(image_resized_pt, disparity_factor)

        # Build intrinsics for unprojection
        intrinsics = (
            torch.tensor(
                [
                    [f_px, 0, width / 2, 0],
                    [0, f_px, height / 2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            .float()
            .to(device)
        )
        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= internal_shape[0] / width
        intrinsics_resized[1] *= internal_shape[1] / height

        # Convert Gaussians to metric space
        gaussians = unproject_gaussians(
            gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
        )

        return gaussians


NODE_CLASS_MAPPINGS = {
    "SharpPredict": SharpPredict,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpPredict": "SHARP Predict (Image to PLY)",
}
