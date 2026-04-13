"""SharpPredictDepth node for ComfyUI-Sharp.

Runs SHARP inference but outputs depth maps instead of PLY files.
Used for panorama depth visualization and verification before Gaussian creation.
"""

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from comfy_api.latest import io
from .utils.image import comfy_to_numpy_rgb

log = logging.getLogger("sharp")


class SharpPredictDepth(io.ComfyNode):
    """Run SHARP inference to generate depth maps from images.

    Unlike SharpPredict which outputs PLY files, this node outputs
    depth maps as images for visualization and further processing.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharpPredictDepth",
            display_name="SHARP Predict Depth",
            category="SHARP",
            description="Generate depth maps from images using SHARP. Optionally align to reference depth (e.g., DepthAnythingV3) using learned dense alignment.",
            inputs=[
                io.Custom("SHARP_MODEL_CONFIG").Input("model"),
                io.Image.Input("image"),
                io.Custom("EXTRINSICS").Input("extrinsics", optional=True,
                                             tooltip="Camera extrinsics (from SamplePanorama). Passed through for pipeline."),
                io.Custom("INTRINSICS").Input("intrinsics", optional=True,
                                             tooltip="Camera intrinsics (from SamplePanorama). Used for depth scale."),
                io.Image.Input("reference_depth", optional=True,
                               tooltip="Reference depth maps (e.g., from DepthAnythingV3) for dense alignment. Must match image batch size."),
            ],
            outputs=[
                io.Image.Output(display_name="depth_maps"),
                io.Custom("EXTRINSICS").Output(display_name="extrinsics"),
                io.Custom("INTRINSICS").Output(display_name="intrinsics"),
                io.Image.Output(display_name="alignment_maps"),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        model,
        image: torch.Tensor,
        extrinsics: torch.Tensor = None,
        intrinsics: torch.Tensor = None,
        reference_depth: torch.Tensor = None,
    ):
        """Run SHARP inference and extract depth maps.

        Returns depth maps as [N, H, W, 1] tensor (grayscale images).
        If reference_depth is provided, uses SHARP's learned dense alignment.
        """
        import comfy.model_management
        from .load_model import _load_sharp_model

        # model is a config dict from LoadSharpModel — load on-demand
        patcher = _load_sharp_model(model)
        predictor = patcher.model
        device = patcher.load_device

        # Handle batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        batch_size = image.shape[0]
        log.info(f"Processing {batch_size} image(s)")

        # Check if we have reference depth for alignment
        use_alignment = reference_depth is not None
        if use_alignment:
            if reference_depth.dim() == 3:
                reference_depth = reference_depth.unsqueeze(0)
            if reference_depth.shape[0] != batch_size:
                log.warning(f"reference_depth batch size ({reference_depth.shape[0]}) "
                            f"doesn't match image batch size ({batch_size}). Disabling alignment.")
                use_alignment = False
            else:
                # Check if scale_map_estimator is available
                scale_map_estimator = predictor.depth_alignment.scale_map_estimator
                if scale_map_estimator is None:
                    log.warning("scale_map_estimator not available in model. Disabling alignment.")
                    use_alignment = False
                else:
                    log.info("Using learned dense alignment with reference depth")

        # SHARP processes at 1536x1536 internally
        # We output depth at native disparity resolution (1536x1536)
        internal_shape = (1536, 1536)

        # Load to GPU with dynamic memory budget based on input shape
        input_shape = [1, 3, internal_shape[0], internal_shape[1]]
        memory_required = patcher.memory_required(input_shape)
        comfy.model_management.load_models_gpu([patcher], memory_required=memory_required)

        all_depth_maps = []
        all_alignment_maps = []

        inference_start = time.time()

        for i in range(batch_size):
            # Extract single image from batch
            single_image = image[i:i+1]
            image_np = comfy_to_numpy_rgb(single_image)
            height, width = image_np.shape[:2]

            if i == 0:
                log.info(f"Image size: {width}x{height}")

            # Get focal length from intrinsics or use default
            if intrinsics is not None:
                f_px = intrinsics[0, 0].item()
            else:
                # Default: assume 65 deg FOV for the input image size
                import math
                fov_rad = math.radians(65)
                f_px = (width / 2) / math.tan(fov_rad / 2)

            # Convert to tensor and normalize
            image_pt = torch.from_numpy(image_np.copy()).float().to(device).permute(2, 0, 1) / 255.0

            # Resize to internal resolution
            image_resized_pt = F.interpolate(
                image_pt[None],
                size=(internal_shape[1], internal_shape[0]),
                mode="bilinear",
                align_corners=True,
            )

            # Run SHARP encode/decode
            log.info(f"Running inference on image {i+1}/{batch_size}...")

            encode_start = time.time()
            monodepth_output, depth_decoder_features = predictor.encode(image_resized_pt)
            encode_time = time.time() - encode_start

            # Compute disparity factor (consistent across all views with same FOV)
            disparity_factor = f_px / width

            log.info(f"  Encode: {encode_time:.2f}s, disparity_factor: {disparity_factor:.4f}")

            # Extract depth from RAW DISPARITY (not Gaussians!)
            # monodepth_output.disparity is [1, 1, 1536, 1536] at internal resolution
            raw_disparity = monodepth_output.disparity  # [1, 1, 1536, 1536]

            # Apply learned dense alignment if reference depth is provided
            if use_alignment:
                # Get reference depth for this image
                ref_depth_single = reference_depth[i]  # [H, W, C]

                # Convert to [1, 1, H, W] and resize to internal resolution
                ref_depth_gray = ref_depth_single[:, :, 0]  # Take first channel
                ref_depth_tensor = ref_depth_gray.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
                ref_depth_resized = F.interpolate(
                    ref_depth_tensor,
                    size=internal_shape,
                    mode="bilinear",
                    align_corners=True,
                )

                # Convert reference depth to same scale as SHARP disparity
                # Reference depth is likely in [0, 1] range, convert to depth-like values
                # Assuming reference is normalized: 1 = close, 0 = far (like our output)
                # Convert to actual depth: larger value = farther
                ref_depth_scaled = 1.0 / (ref_depth_resized.clamp(min=0.01) + 1e-4)

                # SHARP's alignment expects depth (not disparity), so convert
                sharp_depth = disparity_factor / raw_disparity.clamp(min=1e-4)

                # Run scale_map_estimator
                # It expects: tensor_src (SHARP depth), tensor_tgt (reference depth)
                alignment_map = scale_map_estimator(
                    sharp_depth,
                    ref_depth_scaled,
                    depth_decoder_features if predictor.depth_alignment.scale_map_estimator.depth_decoder_features else None,
                )

                # Apply alignment: aligned_depth = alignment_map * sharp_depth
                aligned_depth = alignment_map * sharp_depth

                # Convert back to disparity for consistent output
                aligned_disparity = disparity_factor / aligned_depth.clamp(min=1e-4)

                log.info(f"  Alignment map range: {alignment_map.min():.3f} - {alignment_map.max():.3f}")

                # Store alignment map for visualization
                alignment_map_vis = alignment_map[0, 0].cpu()  # [H, W]
                # Normalize for visualization
                align_min, align_max = alignment_map_vis.min(), alignment_map_vis.max()
                if align_max > align_min:
                    alignment_map_vis = (alignment_map_vis - align_min) / (align_max - align_min)
                alignment_map_rgb = alignment_map_vis.unsqueeze(-1).repeat(1, 1, 3)
                all_alignment_maps.append(alignment_map_rgb)

                # Use aligned disparity
                disparity_resized = aligned_disparity[0, 0]  # [1536, 1536]
            else:
                # No alignment - use raw disparity
                disparity_resized = raw_disparity[0, 0]  # [1536, 1536]

                # Create dummy alignment map (all ones)
                alignment_map_rgb = torch.ones(internal_shape[0], internal_shape[1], 3)
                all_alignment_maps.append(alignment_map_rgb)

            # Convert disparity to depth: depth = disparity_factor / disparity
            depth_map_raw = disparity_factor / disparity_resized.clamp(min=1e-4)  # [H, W]

            if i == 0:
                log.debug(f"  Depth range: {depth_map_raw.min():.3f} - {depth_map_raw.max():.3f}")

            # Convert to [H, W, 1] format for ComfyUI
            depth_map = depth_map_raw.unsqueeze(-1).cpu()  # [H, W, 1]

            # Store as single channel (will be converted to RGB later if needed)
            depth_map_rgb = depth_map.repeat(1, 1, 3)  # [H, W, 3]

            all_depth_maps.append(depth_map_rgb)

        inference_time = time.time() - inference_start
        log.info(f"Total inference time: {inference_time:.2f}s")

        # Stack all depth maps
        depth_maps_batch = torch.stack(all_depth_maps, dim=0)  # [N, H, W, 3]
        alignment_maps_batch = torch.stack(all_alignment_maps, dim=0)  # [N, H, W, 3]

        log.info(f"Output shape: {depth_maps_batch.shape}")

        # Scale intrinsics to match output depth map size (1536x1536 internal resolution)
        # Input intrinsics are for the original image size
        if intrinsics is not None:
            # Get scale factor: output_size / input_size
            input_size = width  # Original image width
            output_size = internal_shape[0]  # 1536 (native disparity resolution)
            scale_factor = output_size / input_size

            if abs(scale_factor - 1.0) > 0.01:  # Only scale if significantly different
                log.info(f"Scaling intrinsics by {scale_factor:.3f} "
                         f"(input {input_size} -> output {output_size})")
                # Scale f_px, f_py, cx, cy
                intrinsics_scaled = intrinsics.clone()
                intrinsics_scaled[0, 0] *= scale_factor  # f_px
                intrinsics_scaled[1, 1] *= scale_factor  # f_py
                intrinsics_scaled[0, 2] *= scale_factor  # cx
                intrinsics_scaled[1, 2] *= scale_factor  # cy
                intrinsics = intrinsics_scaled

        return io.NodeOutput(depth_maps_batch, extrinsics, intrinsics, alignment_maps_batch)


NODE_CLASS_MAPPINGS = {
    "SharpPredictDepth": SharpPredictDepth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpPredictDepth": "SHARP Predict Depth",
}
