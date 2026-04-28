"""SharpPredict node for ComfyUI-Sharp."""

import hashlib
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from comfy_api.latest import io

log = logging.getLogger("sharp")

# Try to import ComfyUI folder_paths for output directory
try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

from .utils.image import comfy_to_numpy_rgb, convert_focallength


# Global cache for encoded features (single image only)
_encode_cache = {
    "image_hash": None,
    "monodepth_output": None,
    "image_resized": None,
    "original_shape": None,
}


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


def _compute_image_hash(image_np: np.ndarray) -> str:
    """Compute hash of image for cache key."""
    return hashlib.sha256(image_np.tobytes()).hexdigest()[:16]


class SharpPredict(io.ComfyNode):
    """Run SHARP inference to generate 3D Gaussians from a single image or batch."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharpPredict",
            display_name="SHARP Predict (Image to PLY)",
            category="SHARP",
            description="Generate 3D Gaussian Splatting PLY file(s) from image(s) using SHARP. Batch input creates a folder with numbered PLY files.",
            is_output_node=True,
            inputs=[
                io.Custom("SHARP_MODEL_CONFIG").Input("model"),
                io.Image.Input("image"),
                io.Float.Input("focal_length_mm", default=30.0, min=0.0, max=500.0, step=0.1,
                               optional=True,
                               tooltip="Focal length in mm (35mm equivalent). 0 = auto (defaults to 30mm). Ignored if intrinsics provided."),
                io.String.Input("output_prefix", default="sharp", optional=True,
                                tooltip="Prefix for output PLY filename or folder name for batches."),
                io.Custom("EXTRINSICS").Input("extrinsics", optional=True,
                                             tooltip="Camera extrinsics (from SamplePanorama). If batched, must match image batch size."),
                io.Custom("INTRINSICS").Input("intrinsics", optional=True,
                                             tooltip="Camera intrinsics (from SamplePanorama). Overrides focal_length_mm if provided."),
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
        output_prefix: str = "sharp",
        extrinsics: torch.Tensor = None,
        intrinsics: torch.Tensor = None,
    ):
        """Run SHARP inference and save PLY file(s).

        For single image: saves {prefix}_{timestamp}.ply
        For batch: creates folder {prefix}_{timestamp}/ with 001.ply, 002.ply, etc.

        Features are cached per image - changing focal_length with same image is instant.

        If extrinsics/intrinsics are provided (from SamplePanorama), Gaussians are
        unprojected into world coordinates using those camera parameters.
        """
        import comfy.model_management
        import comfy.utils
        from .load_model import _load_sharp_model
        from .sharp.gaussians import save_ply, unproject_gaussians

        # model is a config dict from LoadSharpModel — load on-demand
        patcher = _load_sharp_model(model)
        predictor = patcher.model
        device = patcher.load_device

        # Handle batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        batch_size = image.shape[0]

        # Validate extrinsics batch size if provided
        has_camera_params = extrinsics is not None and intrinsics is not None
        if has_camera_params:
            if extrinsics.dim() == 2:
                extrinsics = extrinsics.unsqueeze(0)
            if extrinsics.shape[0] != batch_size:
                raise ValueError(f"Extrinsics batch size ({extrinsics.shape[0]}) must match image batch size ({batch_size})")
            log.info(f"Processing {batch_size} image(s) with provided camera parameters (panorama mode)")
        else:
            log.info(f"Processing {batch_size} image(s)")

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)

        # Determine output path(s)
        if batch_size == 1:
            # Single image: save directly as PLY file
            output_filename = f"{output_prefix}_{timestamp}.ply"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            is_batch = False
        else:
            # Multiple images: create folder
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
            # Extract single image from batch
            single_image = image[i:i+1]
            image_np = comfy_to_numpy_rgb(single_image)
            height, width = image_np.shape[:2]

            if i == 0:
                log.info(f"Image size: {width}x{height}")

            # Get camera parameters for this image
            if has_camera_params:
                # Use provided intrinsics (extract focal length)
                img_intrinsics = intrinsics.to(device)
                img_extrinsics = extrinsics[i].to(device)
                f_px = img_intrinsics[0, 0].item()  # fx from intrinsics matrix
            else:
                # Use focal_length_mm parameter
                if focal_length_mm > 0:
                    f_px = convert_focallength(width, height, focal_length_mm)
                else:
                    f_px = convert_focallength(width, height, 30.0)
                img_extrinsics = None
                img_intrinsics = None

            # Run inference with caching
            log.info(f"Running inference on image {i+1}/{batch_size}...")
            gaussians = _predict_image_cached(
                patcher, predictor, image_np, f_px, device,
                extrinsics=img_extrinsics,
                intrinsics=img_intrinsics,
            )

            # Determine output filename
            if is_batch:
                ply_filename = f"{i+1:03d}.ply"
                ply_path = os.path.join(output_folder, ply_filename)
            else:
                ply_path = output_path

            # Save PLY and get metadata
            _, metadata = save_ply(gaussians, f_px, (height, width), Path(ply_path))

            all_ply_paths.append(ply_path)
            all_extrinsics.append(metadata["extrinsic"])
            all_intrinsics.append(metadata["intrinsic"])

            log.info(f"Saved: {ply_path} ({metadata['num_gaussians']:,} gaussians)")
            pbar.update(1)

        inference_time = time.time() - inference_start
        log.info(f"Total inference time: {inference_time:.2f}s ({inference_time/batch_size:.2f}s per image)")

        # Return values
        if is_batch:
            # For batch: return folder path, and first image's camera params
            # (assuming all images have same camera - user can override)
            return io.NodeOutput(output_path, all_extrinsics[0], all_intrinsics[0])
        else:
            # For single image: return PLY path and camera params
            return io.NodeOutput(output_path, all_extrinsics[0], all_intrinsics[0])


def _predict_image_cached(
    patcher,
    predictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    extrinsics: torch.Tensor = None,
    intrinsics: torch.Tensor = None,
):
    """Predict Gaussians with caching of encoded features.

    The expensive encode step is cached per image.
    Changing focal_length reuses cached features (instant).

    Args:
        patcher: ComfyUI ModelPatcher wrapping the predictor
        predictor: SHARP predictor model
        image: Input image as numpy array [H, W, 3]
        f_px: Focal length in pixels
        device: Torch device
        extrinsics: Optional 4x4 camera extrinsics (world-to-camera)
        intrinsics: Optional 4x4 camera intrinsics
    """
    global _encode_cache
    import comfy.model_management
    from .sharp.gaussians import unproject_gaussians

    internal_shape = (1536, 1536)

    # Load to GPU with dynamic memory budget based on input shape
    input_shape = [1, 3, internal_shape[0], internal_shape[1]]
    memory_required = patcher.memory_required(input_shape)
    comfy.model_management.load_models_gpu([patcher], memory_required=memory_required)
    height, width = image.shape[:2]

    # Compute image hash for cache
    image_hash = _compute_image_hash(image)

    # Check cache
    if _encode_cache["image_hash"] == image_hash:
        # Cache hit - reuse encoded features
        log.info("Cache hit - reusing encoded features (focal_length change is instant)")
        monodepth_output = _monodepth_to(_encode_cache["monodepth_output"], device)
        image_resized_pt = _encode_cache["image_resized"].to(device)
    else:
        # Cache miss - need to encode
        log.info("Encoding...")

        # Clear old cache
        _encode_cache["image_hash"] = None
        _encode_cache["monodepth_output"] = None
        _encode_cache["image_resized"] = None
        _encode_cache["original_shape"] = None

        # Convert to tensor and normalize
        image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0

        # Resize to internal resolution
        image_resized_pt = F.interpolate(
            image_pt[None],
            size=(internal_shape[1], internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )

        # Encode
        encode_start = time.time()
        monodepth_output, _ = predictor.encode(image_resized_pt)
        log.info(f"Encode time: {time.time() - encode_start:.2f}s")

        # Update cache (store on CPU to free VRAM)
        _encode_cache["image_hash"] = image_hash
        _encode_cache["monodepth_output"] = _monodepth_to(monodepth_output, "cpu")
        _encode_cache["image_resized"] = image_resized_pt.cpu()
        _encode_cache["original_shape"] = (height, width)

        # Release fragmented GPU memory after heavy encode (35 ViT passes)
        comfy.model_management.soft_empty_cache()

    # Decode - always run with current focal length
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    decode_start = time.time()
    gaussians_ndc = predictor.decode(monodepth_output, image_resized_pt, disparity_factor)
    log.info(f"Decode time: {time.time() - decode_start:.2f}s")

    # Build intrinsics for unprojection (use provided or construct from f_px)
    if intrinsics is not None:
        unproj_intrinsics = intrinsics.clone()
    else:
        unproj_intrinsics = (
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

    # Scale intrinsics to internal resolution
    intrinsics_resized = unproj_intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Use provided extrinsics or identity
    if extrinsics is not None:
        unproj_extrinsics = extrinsics
    else:
        unproj_extrinsics = torch.eye(4, device=device)

    # Convert Gaussians to world/metric space
    gaussians = unproject_gaussians(
        gaussians_ndc, unproj_extrinsics, intrinsics_resized, internal_shape
    )

    return gaussians


NODE_CLASS_MAPPINGS = {
    "SharpPredict": SharpPredict,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpPredict": "SHARP Predict (Image to PLY)",
}
