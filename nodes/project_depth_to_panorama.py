"""ProjectDepthToPanorama node for ComfyUI-Sharp.

Projects depth maps from perspective views back to equirectangular panorama
with blending and debug visualization.
"""

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from comfy_api.latest import io

log = logging.getLogger("sharp")


def extrinsics_to_yaw_pitch(extrinsics: torch.Tensor) -> tuple[float, float]:
    """Extract yaw and pitch from extrinsics matrix.

    The extrinsics is world-to-camera, so we need the inverse (camera-to-world)
    to get the camera's viewing direction.
    """
    # Camera-to-world is inverse of world-to-camera
    # For pure rotation, inverse = transpose
    R_c2w = extrinsics[:3, :3].T

    # Camera looks along +Z in camera space
    # Transform to world space
    forward = R_c2w @ torch.tensor([0., 0., 1.], device=extrinsics.device)

    # Extract yaw (rotation around Y) and pitch (rotation around X)
    yaw = math.atan2(forward[0].item(), forward[2].item())
    pitch = math.asin(torch.clamp(forward[1], -1, 1).item())

    return yaw, pitch


def compute_blend_weight(
    uu: torch.Tensor,
    vv: torch.Tensor,
    depth_w: int,
    depth_h: int,
    half_fov: float,
    blend_mode: str,
) -> torch.Tensor:
    """Compute blending weights based on distance from view center.

    Args:
        uu, vv: Pixel coordinate grids
        depth_w, depth_h: Depth map dimensions
        half_fov: Half field of view in radians
        blend_mode: Blending algorithm to use

    Returns:
        Weight tensor [H, W]
    """
    # Normalized distance from center (0 at center, 1 at edge)
    dist_from_center = torch.sqrt(
        ((uu - depth_w / 2) / (depth_w / 2)) ** 2 +
        ((vv - depth_h / 2) / (depth_h / 2)) ** 2
    )

    if blend_mode == "cosine":
        # Cosine falloff (original)
        angle_from_center = dist_from_center * half_fov
        weight = torch.cos(torch.clamp(angle_from_center, 0, half_fov))

    elif blend_mode == "gaussian":
        # Gaussian falloff (smoother)
        sigma = 0.5  # Standard deviation in normalized units
        weight = torch.exp(-0.5 * (dist_from_center / sigma) ** 2)

    elif blend_mode == "linear":
        # Linear falloff
        weight = 1.0 - torch.clamp(dist_from_center, 0, 1)

    elif blend_mode == "quadratic":
        # Quadratic falloff (smoother than linear)
        weight = 1.0 - torch.clamp(dist_from_center, 0, 1) ** 2

    elif blend_mode == "feather":
        # Hard center with feathered edges (feather starts at 70% from center)
        feather_start = 0.7
        feather_width = 1.0 - feather_start
        weight = torch.where(
            dist_from_center < feather_start,
            torch.ones_like(dist_from_center),
            1.0 - (dist_from_center - feather_start) / feather_width
        )

    else:  # "none" or fallback
        weight = torch.ones_like(dist_from_center)

    return torch.clamp(weight, 0.01, 1.0)


def project_depth_maps_to_panorama_with_disagreement(
    depth_maps: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    output_width: int,
    blend_mode: str = "gaussian",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project depth maps back to equirectangular panorama with disagreement tracking.

    Args:
        depth_maps: [N, H, W, C] depth maps from each view
        extrinsics: [N, 4, 4] camera extrinsics for each view
        intrinsics: [4, 4] shared camera intrinsics
        output_width: Width of output panorama (height = width/2)
        blend_mode: Blending algorithm ("none", "cosine", "gaussian", "linear", "quadratic", "feather")

    Returns:
        Tuple of:
        - Panoramic depth map [output_height, output_width, 1]
        - Disagreement map [output_height, output_width, 1] (std dev of depths)
        - Overlap count [output_height, output_width] (number of views per pixel)
    """
    device = depth_maps.device
    num_views = depth_maps.shape[0]
    depth_h, depth_w = depth_maps.shape[1:3]
    output_height = output_width // 2

    # Get FOV from intrinsics
    f_px = intrinsics[0, 0].item()
    cx = intrinsics[0, 2].item()
    half_fov = math.atan((depth_w / 2) / f_px)

    # Create output panorama accumulators
    # For Welford's online variance algorithm
    panorama_depth = torch.zeros(output_height, output_width, device=device)  # weighted mean
    panorama_weight = torch.zeros(output_height, output_width, device=device)
    panorama_m2 = torch.zeros(output_height, output_width, device=device)  # sum of squared differences
    panorama_count = torch.zeros(output_height, output_width, device=device)  # view count

    # Store per-view contributions for disagreement calculation
    # We'll use a simpler approach: track sum, sum of squares, and count
    panorama_sum = torch.zeros(output_height, output_width, device=device)
    panorama_sum_sq = torch.zeros(output_height, output_width, device=device)

    # For each view, project its depth to panorama
    for view_idx in range(num_views):
        depth_map = depth_maps[view_idx, :, :, 0]  # [H, W]
        ext = extrinsics[view_idx]

        # Get view direction
        view_yaw, view_pitch = extrinsics_to_yaw_pitch(ext)

        # Create pixel grid for depth map
        u = torch.arange(depth_w, dtype=torch.float32, device=device)
        v = torch.arange(depth_h, dtype=torch.float32, device=device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')

        # Convert to camera-space ray directions
        dx = (uu - cx) / f_px
        dy = (vv - cx) / f_px  # Assuming square pixels, cy = cx
        dz = torch.ones_like(dx)

        rays_cam = torch.stack([dx, dy, dz], dim=-1)
        rays_cam = F.normalize(rays_cam, dim=-1)

        # Rotate to world space (camera-to-world)
        R_c2w = ext[:3, :3].T
        rays_world = torch.einsum('ij,hwj->hwi', R_c2w, rays_cam)

        # Convert world rays to spherical (yaw, pitch)
        rx, ry, rz = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]
        ray_yaw = torch.atan2(rx, rz)
        ray_pitch = torch.asin(torch.clamp(ry, -1, 1))

        # Map to panorama pixel coordinates
        pano_x = ((ray_yaw / math.pi + 1) * (output_width - 1) / 2).long()
        pano_y = ((0.5 - ray_pitch / math.pi) * (output_height - 1)).long()

        # Clamp to valid range
        pano_x = torch.clamp(pano_x, 0, output_width - 1)
        pano_y = torch.clamp(pano_y, 0, output_height - 1)

        # Compute weight for blending
        weight = compute_blend_weight(uu, vv, depth_w, depth_h, half_fov, blend_mode)

        # Scatter depth values to panorama
        flat_pano_idx = pano_y * output_width + pano_x
        flat_depth = depth_map.flatten()
        flat_weight = weight.flatten()

        # Accumulate weighted depth
        panorama_depth.view(-1).scatter_add_(0, flat_pano_idx.flatten(), flat_depth * flat_weight)
        panorama_weight.view(-1).scatter_add_(0, flat_pano_idx.flatten(), flat_weight.flatten())

        # Accumulate for variance calculation (unweighted for simplicity)
        panorama_sum.view(-1).scatter_add_(0, flat_pano_idx.flatten(), flat_depth)
        panorama_sum_sq.view(-1).scatter_add_(0, flat_pano_idx.flatten(), flat_depth ** 2)
        panorama_count.view(-1).scatter_add_(0, flat_pano_idx.flatten(), torch.ones_like(flat_depth))

    # Normalize by accumulated weights
    panorama_depth = panorama_depth / (panorama_weight + 1e-8)

    # Compute variance: var = E[X^2] - E[X]^2
    # Only meaningful where count >= 2
    mean_depth = panorama_sum / (panorama_count + 1e-8)
    mean_sq_depth = panorama_sum_sq / (panorama_count + 1e-8)
    variance = mean_sq_depth - mean_depth ** 2
    variance = torch.clamp(variance, min=0)  # Numerical stability
    std_dev = torch.sqrt(variance)

    # Compute relative disagreement (normalized by mean depth)
    relative_disagreement = std_dev / (mean_depth + 1e-8)

    # Print statistics
    overlap_mask = panorama_count >= 2
    if overlap_mask.any():
        overlap_std = std_dev[overlap_mask]
        overlap_rel = relative_disagreement[overlap_mask]
        log.info(f"Overlap regions ({overlap_mask.sum().item()} pixels):")
        log.info(f"    Absolute disagreement: mean={overlap_std.mean():.4f}, max={overlap_std.max():.4f}")
        log.info(f"    Relative disagreement: mean={overlap_rel.mean():.2%}, max={overlap_rel.max():.2%}")

        # Count pixels with high disagreement (>10% relative)
        high_disagreement = (relative_disagreement > 0.1) & overlap_mask
        log.info(f"    High disagreement (>10%): {high_disagreement.sum().item()} pixels "
                 f"({high_disagreement.sum().item() / overlap_mask.sum().item() * 100:.1f}% of overlap)")

    # Global normalization for visualization
    valid_mask = panorama_weight > 0.01
    depth_min, depth_max = 0.0, 1.0
    if valid_mask.any():
        valid_depths = panorama_depth[valid_mask]
        depth_min = valid_depths.min()
        depth_max = valid_depths.max()
        log.info(f"Depth range: {depth_min:.4f} to {depth_max:.4f}")

        if depth_max > depth_min:
            panorama_depth = torch.where(
                valid_mask,
                1.0 - (panorama_depth - depth_min) / (depth_max - depth_min),
                torch.zeros_like(panorama_depth)
            )
        else:
            panorama_depth = torch.where(
                valid_mask,
                torch.full_like(panorama_depth, 0.5),
                torch.zeros_like(panorama_depth)
            )

    return (
        panorama_depth.unsqueeze(-1),
        relative_disagreement.unsqueeze(-1),
        panorama_count,
    )


def project_depth_maps_to_panorama(
    depth_maps: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    output_width: int,
    blend_mode: str = "gaussian",
) -> torch.Tensor:
    """Project depth maps back to equirectangular panorama.

    Args:
        depth_maps: [N, H, W, C] depth maps from each view
        extrinsics: [N, 4, 4] camera extrinsics for each view
        intrinsics: [4, 4] shared camera intrinsics
        output_width: Width of output panorama (height = width/2)
        blend_mode: Blending algorithm ("none", "cosine", "gaussian", "linear", "quadratic", "feather")

    Returns:
        Panoramic depth map [output_height, output_width, 1]
    """
    depth, _, _ = project_depth_maps_to_panorama_with_disagreement(
        depth_maps, extrinsics, intrinsics, output_width, blend_mode
    )
    return depth


def get_disagreement_color(d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert normalized disagreement [0,1] to RGB color.

    Color gradient: blue (0) -> cyan (0.25) -> green (0.5) -> yellow (0.75) -> red (1.0)
    """
    # Red channel: ramps up from 0.5 to 1.0
    r = torch.where(d < 0.5, 2 * d, torch.ones_like(d))

    # Green channel: ramps up from 0 to 0.5, then down from 0.5 to 1.0
    g = torch.where(d < 0.5, 2 * d, 2 * (1 - d))

    # Blue channel: ramps down from 0 to 0.5
    b = torch.where(d < 0.5, 1 - 2 * d, torch.zeros_like(d))

    return r, g, b


def draw_legend(
    heatmap: torch.Tensor,
    max_disagreement: float,
    legend_width: int = 30,
    margin: int = 10,
) -> torch.Tensor:
    """Draw a color legend bar on the right side of the heatmap.

    Args:
        heatmap: [H, W, 3] input heatmap
        max_disagreement: Maximum disagreement value for scale
        legend_width: Width of the legend bar in pixels
        margin: Margin around the legend

    Returns:
        [H, W + legend_width + margin*2, 3] heatmap with legend
    """
    device = heatmap.device
    H, W = heatmap.shape[:2]

    # Create extended canvas
    total_width = W + legend_width + margin * 3
    canvas = torch.zeros(H, total_width, 3, device=device)

    # Copy original heatmap
    canvas[:, :W, :] = heatmap

    # Legend area
    legend_x_start = W + margin
    legend_x_end = legend_x_start + legend_width

    # Color bar region (leave space for labels at top and bottom)
    label_height = 40
    bar_y_start = margin + label_height
    bar_y_end = H - margin - label_height
    bar_height = bar_y_end - bar_y_start

    if bar_height > 0:
        # Draw gradient color bar
        for y in range(bar_y_start, bar_y_end):
            # Map y to disagreement value (top = max, bottom = 0)
            t = 1.0 - (y - bar_y_start) / bar_height
            d = torch.tensor([t], device=device)
            r, g, b = get_disagreement_color(d)

            canvas[y, legend_x_start:legend_x_end, 0] = r.item()
            canvas[y, legend_x_start:legend_x_end, 1] = g.item()
            canvas[y, legend_x_start:legend_x_end, 2] = b.item()

        # Draw border around color bar
        canvas[bar_y_start:bar_y_end, legend_x_start, :] = 1.0  # Left
        canvas[bar_y_start:bar_y_end, legend_x_end-1, :] = 1.0  # Right
        canvas[bar_y_start, legend_x_start:legend_x_end, :] = 1.0  # Top
        canvas[bar_y_end-1, legend_x_start:legend_x_end, :] = 1.0  # Bottom

        # Draw tick marks and approximate label positions
        # We'll draw simple markers since we can't render text
        tick_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        tick_width = 5

        for t in tick_positions:
            y = int(bar_y_start + (1 - t) * (bar_height - 1))
            # Draw tick mark
            canvas[y, legend_x_end:legend_x_end + tick_width, :] = 1.0

        # Draw "single view" indicator (dark gray square)
        indicator_y = H - margin - 20
        indicator_size = 15
        canvas[indicator_y:indicator_y+indicator_size,
               legend_x_start:legend_x_start+indicator_size, :] = 0.2
        # Border
        canvas[indicator_y, legend_x_start:legend_x_start+indicator_size, :] = 0.5
        canvas[indicator_y+indicator_size-1, legend_x_start:legend_x_start+indicator_size, :] = 0.5
        canvas[indicator_y:indicator_y+indicator_size, legend_x_start, :] = 0.5
        canvas[indicator_y:indicator_y+indicator_size, legend_x_start+indicator_size-1, :] = 0.5

    # Add text labels using simple block letters
    # Top label: max value (e.g., "20%")
    pct = int(max_disagreement * 100)
    draw_text_simple(canvas, f"{pct}%", legend_x_start, margin + 5, device)
    draw_text_simple(canvas, "BAD", legend_x_start, margin + 20, device)

    # Bottom label
    draw_text_simple(canvas, "0%", legend_x_start, bar_y_end + 5, device)
    draw_text_simple(canvas, "GOOD", legend_x_start, bar_y_end + 20, device)

    # Single view label
    draw_text_simple(canvas, "1 VIEW", legend_x_start + 18, indicator_y + 3, device)

    return canvas


def draw_text_simple(canvas: torch.Tensor, text: str, x: int, y: int, device: torch.device):
    """Draw simple blocky text on canvas. Very basic 3x5 pixel font."""
    # Simple 3x5 pixel font for basic characters
    font = {
        '0': [[1,1,1], [1,0,1], [1,0,1], [1,0,1], [1,1,1]],
        '1': [[0,1,0], [1,1,0], [0,1,0], [0,1,0], [1,1,1]],
        '2': [[1,1,1], [0,0,1], [1,1,1], [1,0,0], [1,1,1]],
        '3': [[1,1,1], [0,0,1], [1,1,1], [0,0,1], [1,1,1]],
        '4': [[1,0,1], [1,0,1], [1,1,1], [0,0,1], [0,0,1]],
        '5': [[1,1,1], [1,0,0], [1,1,1], [0,0,1], [1,1,1]],
        '6': [[1,1,1], [1,0,0], [1,1,1], [1,0,1], [1,1,1]],
        '7': [[1,1,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1]],
        '8': [[1,1,1], [1,0,1], [1,1,1], [1,0,1], [1,1,1]],
        '9': [[1,1,1], [1,0,1], [1,1,1], [0,0,1], [1,1,1]],
        '%': [[1,0,1], [0,0,1], [0,1,0], [1,0,0], [1,0,1]],
        'B': [[1,1,0], [1,0,1], [1,1,0], [1,0,1], [1,1,0]],
        'A': [[0,1,0], [1,0,1], [1,1,1], [1,0,1], [1,0,1]],
        'D': [[1,1,0], [1,0,1], [1,0,1], [1,0,1], [1,1,0]],
        'G': [[1,1,1], [1,0,0], [1,0,1], [1,0,1], [1,1,1]],
        'O': [[1,1,1], [1,0,1], [1,0,1], [1,0,1], [1,1,1]],
        'V': [[1,0,1], [1,0,1], [1,0,1], [0,1,0], [0,1,0]],
        'I': [[1,1,1], [0,1,0], [0,1,0], [0,1,0], [1,1,1]],
        'E': [[1,1,1], [1,0,0], [1,1,0], [1,0,0], [1,1,1]],
        'W': [[1,0,1], [1,0,1], [1,1,1], [1,1,1], [1,0,1]],
        ' ': [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
    }

    H, W = canvas.shape[:2]
    cursor_x = x
    scale = 2  # Scale up the tiny font

    for char in text.upper():
        if char in font:
            glyph = font[char]
            for row_idx, row in enumerate(glyph):
                for col_idx, pixel in enumerate(row):
                    if pixel:
                        px = cursor_x + col_idx * scale
                        py = y + row_idx * scale
                        # Draw scaled pixel
                        for dy in range(scale):
                            for dx in range(scale):
                                if 0 <= py + dy < H and 0 <= px + dx < W:
                                    canvas[py + dy, px + dx, :] = 1.0
            cursor_x += (len(glyph[0]) + 1) * scale


def create_disagreement_heatmap(
    disagreement: torch.Tensor,
    overlap_count: torch.Tensor,
    max_disagreement: float = 0.2,
    add_legend: bool = True,
) -> torch.Tensor:
    """Create a color heatmap showing depth disagreement in overlap regions.

    Args:
        disagreement: [H, W, 1] relative disagreement (std/mean)
        overlap_count: [H, W] number of views per pixel
        max_disagreement: Maximum disagreement for color scale (clips above)
        add_legend: Whether to add a color legend bar

    Returns:
        [H, W, 3] or [H, W+legend_width, 3] RGB heatmap where:
        - Black = no overlap (single view or no coverage)
        - Blue = low disagreement (good alignment)
        - Green = medium disagreement
        - Yellow = high disagreement
        - Red = very high disagreement (bad alignment)
    """
    device = disagreement.device
    H, W = disagreement.shape[:2]

    # Normalize disagreement to [0, 1]
    disagreement_norm = (disagreement[:, :, 0] / max_disagreement).clamp(0, 1)

    # Create RGB output
    heatmap = torch.zeros(H, W, 3, device=device)

    # Mask: only show overlap regions (2+ views)
    overlap_mask = overlap_count >= 2

    # Get colors
    r, g, b = get_disagreement_color(disagreement_norm)

    # Apply mask - only show in overlap regions
    heatmap[:, :, 0] = torch.where(overlap_mask, r, torch.zeros_like(r))
    heatmap[:, :, 1] = torch.where(overlap_mask, g, torch.zeros_like(g))
    heatmap[:, :, 2] = torch.where(overlap_mask, b, torch.zeros_like(b))

    # Make non-overlap regions dark gray so you can see coverage
    non_overlap = (overlap_count == 1)
    heatmap[:, :, 0] = torch.where(non_overlap, torch.full_like(r, 0.2), heatmap[:, :, 0])
    heatmap[:, :, 1] = torch.where(non_overlap, torch.full_like(g, 0.2), heatmap[:, :, 1])
    heatmap[:, :, 2] = torch.where(non_overlap, torch.full_like(b, 0.2), heatmap[:, :, 2])

    # Add legend
    if add_legend:
        heatmap = draw_legend(heatmap, max_disagreement)

    return heatmap


def draw_sample_borders(
    panorama: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    border_color: tuple = (1.0, 0.0, 0.0),
    border_width: int = 2,
) -> torch.Tensor:
    """Draw borders around sample regions on the panorama.

    Args:
        panorama: [H, W, C] panorama image
        extrinsics: [N, 4, 4] camera extrinsics for each view
        intrinsics: [4, 4] shared camera intrinsics
        border_color: RGB color for borders
        border_width: Width of border lines in pixels

    Returns:
        Panorama with borders drawn [H, W, 3]
    """
    device = panorama.device
    H, W = panorama.shape[:2]
    num_views = extrinsics.shape[0]

    # Convert grayscale to RGB if needed
    if panorama.shape[-1] == 1:
        panorama_rgb = panorama.repeat(1, 1, 3)
    else:
        panorama_rgb = panorama.clone()

    # Get FOV from intrinsics
    # The intrinsics encode the image size via principal point (cx = width/2)
    f_px = intrinsics[0, 0].item()
    cx = intrinsics[0, 2].item()
    depth_size = int(cx * 2)  # Derive from principal point (cx = width/2)
    half_fov = math.atan((depth_size / 2) / f_px)

    border_color_t = torch.tensor(border_color, device=device)

    for view_idx in range(num_views):
        ext = extrinsics[view_idx]
        view_yaw, view_pitch = extrinsics_to_yaw_pitch(ext)

        # Draw border of this view's coverage area
        # The view covers a square region of angular size 2*half_fov

        # Sample points along the border
        num_border_points = 100

        for edge in ['top', 'bottom', 'left', 'right']:
            if edge == 'top':
                us = torch.linspace(-1, 1, num_border_points)
                vs = torch.ones(num_border_points) * -1
            elif edge == 'bottom':
                us = torch.linspace(-1, 1, num_border_points)
                vs = torch.ones(num_border_points) * 1
            elif edge == 'left':
                us = torch.ones(num_border_points) * -1
                vs = torch.linspace(-1, 1, num_border_points)
            else:  # right
                us = torch.ones(num_border_points) * 1
                vs = torch.linspace(-1, 1, num_border_points)

            # Convert to camera ray directions
            dx = us * math.tan(half_fov)
            dy = vs * math.tan(half_fov)
            dz = torch.ones_like(dx)

            rays_cam = torch.stack([dx, dy, dz], dim=-1).to(device)
            rays_cam = F.normalize(rays_cam, dim=-1)

            # Rotate to world space
            R_c2w = ext[:3, :3].T
            rays_world = rays_cam @ R_c2w.T

            # Convert to spherical
            rx, ry, rz = rays_world[:, 0], rays_world[:, 1], rays_world[:, 2]
            ray_yaw = torch.atan2(rx, rz)
            ray_pitch = torch.asin(torch.clamp(ry, -1, 1))

            # Map to panorama pixels
            pano_x = ((ray_yaw / math.pi + 1) * (W - 1) / 2).long()
            pano_y = ((0.5 - ray_pitch / math.pi) * (H - 1)).long()

            # Draw border pixels
            for bw in range(-border_width // 2, border_width // 2 + 1):
                px = torch.clamp(pano_x + bw, 0, W - 1)
                py = torch.clamp(pano_y, 0, H - 1)
                panorama_rgb[py, px] = border_color_t

                px = torch.clamp(pano_x, 0, W - 1)
                py = torch.clamp(pano_y + bw, 0, H - 1)
                panorama_rgb[py, px] = border_color_t

    return panorama_rgb


class ProjectDepthToPanorama(io.ComfyNode):
    """Project depth maps from perspective views back to equirectangular panorama.

    Combines depth maps with blending and optional debug visualization.
    Outputs a disagreement heatmap showing where overlapping views have mismatching depths.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ProjectDepthToPanorama",
            display_name="Project Depth to Panorama",
            category="SHARP",
            description="Project depth maps to panorama. Outputs disagreement heatmap showing where views mismatch (blue=good, red=bad).",
            inputs=[
                io.Image.Input("depth_maps"),
                io.Custom("EXTRINSICS").Input("extrinsics"),
                io.Custom("INTRINSICS").Input("intrinsics"),
                io.Int.Input("panorama_width", default=2048, min=512, max=8192, step=64,
                             optional=True,
                             tooltip="Output panorama width (height = width/2)"),
                io.Combo.Input("blend_mode",
                               options=["gaussian", "cosine", "quadratic", "linear", "feather", "none"],
                               default="gaussian", optional=True,
                               tooltip="Blending algorithm: gaussian (smoothest), cosine, quadratic, linear, feather (hard center), none"),
                io.Boolean.Input("show_borders", default=True, optional=True,
                                 tooltip="Draw borders around sample regions for debugging"),
                io.Combo.Input("border_color",
                               options=["red", "green", "blue", "white", "yellow"],
                               default="red", optional=True,
                               tooltip="Color for sample region borders"),
                io.Float.Input("disagreement_scale", default=0.2, min=0.01, max=1.0, step=0.01,
                               optional=True,
                               tooltip="Max relative disagreement for heatmap color scale (0.2 = 20%)"),
            ],
            outputs=[
                io.Image.Output(display_name="panoramic_depth"),
                io.Image.Output(display_name="debug_overlay"),
                io.Image.Output(display_name="disagreement_heatmap"),
            ],
        )

    @classmethod
    def execute(
        cls,
        depth_maps: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        panorama_width: int = 2048,
        blend_mode: str = "gaussian",
        show_borders: bool = True,
        border_color: str = "red",
        disagreement_scale: float = 0.2,
    ):
        """Project depth maps to panorama with disagreement visualization."""

        log.info(f"Input: {depth_maps.shape[0]} depth maps")
        log.info(f"Output panorama: {panorama_width} x {panorama_width // 2}")
        log.info(f"Blend mode: {blend_mode}")

        # Project depth maps with disagreement tracking
        panoramic_depth, disagreement, overlap_count = project_depth_maps_to_panorama_with_disagreement(
            depth_maps,
            extrinsics,
            intrinsics,
            panorama_width,
            blend_mode,
        )

        # Convert to RGB for output
        panoramic_depth_rgb = panoramic_depth.repeat(1, 1, 3)

        # Create debug overlay with borders
        if show_borders:
            color_map = {
                "red": (1.0, 0.0, 0.0),
                "green": (0.0, 1.0, 0.0),
                "blue": (0.0, 0.0, 1.0),
                "white": (1.0, 1.0, 1.0),
                "yellow": (1.0, 1.0, 0.0),
            }
            debug_overlay = draw_sample_borders(
                panoramic_depth_rgb.clone(),
                extrinsics,
                intrinsics,
                border_color=color_map.get(border_color, (1.0, 0.0, 0.0)),
            )
        else:
            debug_overlay = panoramic_depth_rgb.clone()

        # Create disagreement heatmap
        disagreement_heatmap = create_disagreement_heatmap(
            disagreement,
            overlap_count,
            max_disagreement=disagreement_scale,
        )

        # Add borders to heatmap too for reference
        if show_borders:
            disagreement_heatmap = draw_sample_borders(
                disagreement_heatmap,
                extrinsics,
                intrinsics,
                border_color=(1.0, 1.0, 1.0),  # White borders on heatmap
                border_width=1,
            )

        # Add batch dimension for ComfyUI
        panoramic_depth_out = panoramic_depth_rgb.unsqueeze(0)  # [1, H, W, 3]
        debug_overlay_out = debug_overlay.unsqueeze(0)  # [1, H, W, 3]
        disagreement_out = disagreement_heatmap.unsqueeze(0)  # [1, H, W, 3]

        log.info(f"Output shape: {panoramic_depth_out.shape}")

        return io.NodeOutput(panoramic_depth_out, debug_overlay_out, disagreement_out)


NODE_CLASS_MAPPINGS = {
    "ProjectDepthToPanorama": ProjectDepthToPanorama,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProjectDepthToPanorama": "Project Depth to Panorama",
}
