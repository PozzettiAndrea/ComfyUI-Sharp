"""SamplePanorama node for ComfyUI-Sharp.

Samples perspective cutouts from an equirectangular panorama for use with SHARP.
"""

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from comfy_api.latest import io

log = logging.getLogger("sharp")


def create_rotation_matrix(yaw: float, pitch: float) -> torch.Tensor:
    """Create a rotation matrix from yaw and pitch angles.

    Args:
        yaw: Horizontal angle in radians (0 = forward, positive = right)
        pitch: Vertical angle in radians (0 = horizon, positive = up)

    Returns:
        3x3 rotation matrix (camera-to-world)
    """
    # Rotation around Y axis (yaw - looking left/right)
    cy, sy = math.cos(yaw), math.sin(yaw)
    R_yaw = torch.tensor([
        [cy,  0, sy],
        [0,   1,  0],
        [-sy, 0, cy]
    ], dtype=torch.float32)

    # Rotation around X axis (pitch - looking up/down)
    cp, sp = math.cos(pitch), math.sin(pitch)
    R_pitch = torch.tensor([
        [1,  0,   0],
        [0,  cp, -sp],
        [0,  sp,  cp]
    ], dtype=torch.float32)

    # Combined: first pitch, then yaw
    R = R_yaw @ R_pitch
    return R


def sample_perspective_from_equirectangular(
    panorama: torch.Tensor,
    yaw: float,
    pitch: float,
    fov_radians: float,
    output_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a perspective view from an equirectangular panorama.

    Args:
        panorama: Equirectangular image [H, W, 3] or [B, H, W, 3]
        yaw: Horizontal angle in radians (0 = center, positive = right)
        pitch: Vertical angle in radians (0 = horizon, positive = up)
        fov_radians: Field of view in radians
        output_size: Output image size (square)

    Returns:
        Tuple of (perspective_image, extrinsics_4x4, intrinsics_4x4)
    """
    if panorama.dim() == 4:
        panorama = panorama[0]  # Take first if batched

    H, W, C = panorama.shape
    device = panorama.device

    # Compute focal length in pixels
    f_px = (output_size / 2) / math.tan(fov_radians / 2)

    # Principal point at center
    cx = (output_size - 1) / 2
    cy = (output_size - 1) / 2

    # Create pixel grid for output image
    u = torch.arange(output_size, dtype=torch.float32, device=device)
    v = torch.arange(output_size, dtype=torch.float32, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')  # [H, W]

    # Convert to camera-space ray directions
    # d_cam = [(u - cx) / f, (v - cy) / f, 1]
    dx = (uu - cx) / f_px
    dy = (vv - cy) / f_px
    dz = torch.ones_like(dx)

    # Stack and normalize
    rays_cam = torch.stack([dx, dy, dz], dim=-1)  # [H, W, 3]
    rays_cam = F.normalize(rays_cam, dim=-1)

    # Rotate rays to world space
    R = create_rotation_matrix(yaw, pitch).to(device)  # [3, 3]
    rays_world = torch.einsum('ij,hwj->hwi', R, rays_cam)  # [H, W, 3]

    # Convert world rays to spherical coordinates
    # x = right, y = up, z = forward
    rx, ry, rz = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]

    # Yaw: angle around Y axis (horizontal)
    ray_yaw = torch.atan2(rx, rz)  # [-pi, pi]

    # Pitch: angle from horizon (vertical)
    ray_pitch = torch.asin(torch.clamp(ry, -1, 1))  # [-pi/2, pi/2]

    # Map to equirectangular pixel coordinates
    # Yaw: -pi..pi -> 0..W
    # Pitch: -pi/2..pi/2 -> H..0 (note: inverted, top of image is up)
    eq_x = (ray_yaw / math.pi + 1) * (W - 1) / 2
    eq_y = (0.5 - ray_pitch / math.pi) * (H - 1)

    # Normalize to [-1, 1] for grid_sample
    grid_x = eq_x / (W - 1) * 2 - 1
    grid_y = eq_y / (H - 1) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    # Sample from panorama
    panorama_nchw = panorama.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    sampled = F.grid_sample(
        panorama_nchw,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    perspective = sampled[0].permute(1, 2, 0)  # [H, W, C]

    # Build intrinsics matrix (4x4)
    intrinsics = torch.tensor([
        [f_px, 0,    cx,   0],
        [0,    f_px, cy,   0],
        [0,    0,    1,    0],
        [0,    0,    0,    1],
    ], dtype=torch.float32, device=device)

    # Build extrinsics matrix (4x4, world-to-camera)
    # World-to-camera is the inverse (transpose for rotation)
    R_w2c = R.T
    extrinsics = torch.eye(4, dtype=torch.float32, device=device)
    extrinsics[:3, :3] = R_w2c
    # No translation (camera at origin)

    return perspective, extrinsics, intrinsics


class SamplePanorama(io.ComfyNode):
    """Sample perspective cutouts from an equirectangular panorama.

    Automatically calculates the number of samples needed to cover the full
    360 x 180 panorama based on FOV and overlap settings.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplePanorama",
            display_name="Sample Panorama (Equirect -> Perspective)",
            category="SHARP",
            description="Sample perspective views from a 360 equirectangular panorama for SHARP processing.",
            inputs=[
                io.Image.Input("panorama"),
                io.Float.Input("fov_degrees", default=65.0, min=30.0, max=120.0, step=1.0,
                               tooltip="Field of view in degrees for each perspective cutout"),
                io.Float.Input("overlap_percent", default=10.0, min=0.0, max=50.0, step=1.0,
                               tooltip="Overlap between adjacent samples as percentage of FOV"),
                io.Int.Input("output_size", default=1536, min=256, max=2048, step=64,
                             tooltip="Output resolution for each perspective image (square). 1536 is SHARP's native resolution."),
                io.Boolean.Input("skip_poles", default=True, optional=True,
                                 tooltip="Skip samples pointing straight up/down (often low quality in panoramas)"),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Custom("EXTRINSICS").Output(display_name="extrinsics"),
                io.Custom("INTRINSICS").Output(display_name="intrinsics"),
                io.Int.Output(display_name="num_horizontal"),
                io.Int.Output(display_name="num_vertical"),
            ],
        )

    @classmethod
    def execute(
        cls,
        panorama: torch.Tensor,
        fov_degrees: float = 65.0,
        overlap_percent: float = 10.0,
        output_size: int = 1536,
        skip_poles: bool = True,
    ):
        """Sample perspective cutouts covering the full panorama."""

        fov_radians = math.radians(fov_degrees)

        # Calculate step angle based on overlap
        step_degrees = fov_degrees * (1 - overlap_percent / 100)
        step_radians = math.radians(step_degrees)

        # Calculate number of samples needed
        num_horizontal = math.ceil(360 / step_degrees)

        # For vertical, we cover -90 to +90 (180 total)
        # But if skip_poles, we limit to avoid extreme angles
        if skip_poles:
            # Skip top and bottom 15
            vertical_range = 150  # degrees
            vertical_start = -75  # degrees
        else:
            vertical_range = 180
            vertical_start = -90

        num_vertical = max(1, math.ceil(vertical_range / step_degrees))

        log.info(f"FOV: {fov_degrees}, Overlap: {overlap_percent}%")
        log.info(f"Step angle: {step_degrees:.1f}")
        log.info(f"Samples: {num_horizontal} horizontal x {num_vertical} vertical = {num_horizontal * num_vertical} total")

        # Handle batch dimension
        if panorama.dim() == 3:
            panorama = panorama.unsqueeze(0)

        # Take first image if batch
        pano = panorama[0]  # [H, W, C]

        all_images = []
        all_extrinsics = []
        intrinsics = None

        for v_idx in range(num_vertical):
            # Calculate pitch angle
            pitch_degrees = vertical_start + (v_idx + 0.5) * step_degrees
            pitch_radians = math.radians(pitch_degrees)

            for h_idx in range(num_horizontal):
                # Calculate yaw angle
                yaw_degrees = -180 + (h_idx + 0.5) * step_degrees
                yaw_radians = math.radians(yaw_degrees)

                # Sample perspective view
                perspective, extrinsics, intr = sample_perspective_from_equirectangular(
                    pano,
                    yaw_radians,
                    pitch_radians,
                    fov_radians,
                    output_size,
                )

                all_images.append(perspective)
                all_extrinsics.append(extrinsics)

                if intrinsics is None:
                    intrinsics = intr

        # Stack results
        images_batch = torch.stack(all_images, dim=0)  # [N, H, W, C]
        extrinsics_batch = torch.stack(all_extrinsics, dim=0)  # [N, 4, 4]

        log.info(f"Output shape: {images_batch.shape}")

        return io.NodeOutput(images_batch, extrinsics_batch, intrinsics, num_horizontal, num_vertical)


NODE_CLASS_MAPPINGS = {
    "SamplePanorama": SamplePanorama,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplePanorama": "Sample Panorama (Equirect -> Perspective)",
}
