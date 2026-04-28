"""AlignDepthMaps node for ComfyUI-Sharp.

RANSAC-based scale-shift alignment to make depth maps globally consistent
across multiple perspective views before panorama projection.
"""

import json
import logging
import math
from collections import deque

import torch
import torch.nn.functional as F

from comfy_api.latest import io

log = logging.getLogger("sharp")


def extrinsics_to_direction(extrinsics: torch.Tensor) -> torch.Tensor:
    """Get the viewing direction (forward vector) from extrinsics.

    Args:
        extrinsics: [4, 4] world-to-camera matrix

    Returns:
        [3] normalized direction vector in world space
    """
    # Camera-to-world is inverse of world-to-camera
    # For pure rotation, inverse = transpose
    R_c2w = extrinsics[:3, :3].T

    # Camera looks along +Z in camera space
    forward = R_c2w @ torch.tensor([0., 0., 1.], device=extrinsics.device)
    return forward


def compute_view_overlap(
    ext_i: torch.Tensor,
    ext_j: torch.Tensor,
    half_fov: float,
) -> float:
    """Compute approximate overlap fraction between two views.

    Uses angular distance between view centers as a proxy for overlap.

    Args:
        ext_i, ext_j: [4, 4] extrinsics matrices
        half_fov: Half field of view in radians

    Returns:
        Overlap fraction [0, 1], where 1 = complete overlap
    """
    dir_i = extrinsics_to_direction(ext_i)
    dir_j = extrinsics_to_direction(ext_j)

    # Angular distance between view centers
    dot = torch.clamp(torch.dot(dir_i, dir_j), -1, 1)
    angle = torch.acos(dot).item()

    # If angle < 2*half_fov, there's overlap
    # Overlap fraction decreases linearly from 1 (same direction) to 0 (2*half_fov apart)
    max_angle = 2 * half_fov
    if angle >= max_angle:
        return 0.0
    return 1.0 - angle / max_angle


def compute_overlap_mask(
    depth_h: int,
    depth_w: int,
    ext_i: torch.Tensor,
    ext_j: torch.Tensor,
    intrinsics: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute which pixels in view i overlap with view j and vice versa.

    Returns corresponding pixel coordinates for RANSAC matching.

    Args:
        depth_h, depth_w: Depth map dimensions
        ext_i, ext_j: [4, 4] extrinsics matrices
        intrinsics: [4, 4] intrinsics matrix
        device: Torch device

    Returns:
        Tuple of (mask_i, mask_j, coords_i, coords_j) where:
        - mask_i: [H, W] bool mask of pixels in i that overlap with j
        - mask_j: [H, W] bool mask of pixels in j that overlap with i
        - coords_i: [N, 2] pixel coordinates in view i
        - coords_j: [N, 2] corresponding pixel coordinates in view j
    """
    f_px = intrinsics[0, 0].item()
    cx = intrinsics[0, 2].item()
    cy = intrinsics[1, 2].item()

    # Create pixel grid for view i
    u = torch.arange(depth_w, dtype=torch.float32, device=device)
    v = torch.arange(depth_h, dtype=torch.float32, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')

    # Convert to camera-space ray directions
    dx = (uu - cx) / f_px
    dy = (vv - cy) / f_px
    dz = torch.ones_like(dx)
    rays_cam = torch.stack([dx, dy, dz], dim=-1)
    rays_cam = F.normalize(rays_cam, dim=-1)

    # Rotate to world space using view i's camera-to-world
    R_i_c2w = ext_i[:3, :3].T
    rays_world = torch.einsum('ij,hwj->hwi', R_i_c2w, rays_cam)

    # Transform world rays to view j's camera space
    R_j_w2c = ext_j[:3, :3]  # World-to-camera for view j
    rays_j_cam = torch.einsum('ij,hwj->hwi', R_j_w2c, rays_world)

    # Project to view j's image plane
    # Only valid if z > 0 (in front of camera)
    z_j = rays_j_cam[..., 2]
    valid_z = z_j > 0.01

    # Perspective projection
    x_j = rays_j_cam[..., 0] / (z_j + 1e-6)
    y_j = rays_j_cam[..., 1] / (z_j + 1e-6)

    # Convert to pixel coordinates
    u_j = x_j * f_px + cx
    v_j = y_j * f_px + cy

    # Check if projected point is within view j's image bounds
    in_bounds = (u_j >= 0) & (u_j < depth_w) & (v_j >= 0) & (v_j < depth_h)

    # Combined mask for view i
    mask_i = valid_z & in_bounds

    # Get pixel coordinates for matching
    coords_i_y, coords_i_x = torch.where(mask_i)
    coords_i = torch.stack([coords_i_x, coords_i_y], dim=1)  # [N, 2] as (x, y)

    # Corresponding coordinates in view j
    u_j_valid = u_j[mask_i]
    v_j_valid = v_j[mask_i]
    coords_j = torch.stack([u_j_valid, v_j_valid], dim=1)  # [N, 2] as (x, y)

    # Create mask_j (approximate - mark pixels that have correspondences)
    mask_j = torch.zeros(depth_h, depth_w, dtype=torch.bool, device=device)
    u_j_int = coords_j[:, 0].long().clamp(0, depth_w - 1)
    v_j_int = coords_j[:, 1].long().clamp(0, depth_h - 1)
    mask_j[v_j_int, u_j_int] = True

    return mask_i, mask_j, coords_i, coords_j


def median_scale_alignment(
    depth1: torch.Tensor,
    depth2: torch.Tensor,
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    debug: bool = True,
) -> tuple[float, float, int, dict]:
    """Find scale to align depth2 to depth1 using median ratio.

    Solves: depth1 = scale * depth2 (NO SHIFT - prevents negative depths)

    Args:
        depth1: [H, W] reference depth map
        depth2: [H, W] depth map to align
        coords1: [N, 2] pixel coordinates in depth1 (x, y)
        coords2: [N, 2] corresponding pixel coordinates in depth2 (x, y)
        debug: Whether to print debug info

    Returns:
        Tuple of (scale, shift=0.0, num_valid, debug_info)
    """
    n_points = coords1.shape[0]
    debug_info = {"overlap_pixels": n_points}

    if n_points < 10:
        return 1.0, 0.0, 0, debug_info

    # Sample depth values at corresponding locations
    x1, y1 = coords1[:, 0].long(), coords1[:, 1].long()
    x2, y2 = coords2[:, 0].long().clamp(0, depth2.shape[1] - 1), coords2[:, 1].long().clamp(0, depth2.shape[0] - 1)

    d1 = depth1[y1, x1]
    d2 = depth2[y2, x2]

    # Filter out invalid depths (must be positive and finite)
    valid = (d1 > 0.1) & (d2 > 0.1) & torch.isfinite(d1) & torch.isfinite(d2)
    d1_valid = d1[valid]
    d2_valid = d2[valid]
    n_valid = len(d1_valid)

    debug_info["valid_pixels"] = n_valid
    debug_info["d1_range"] = (d1_valid.min().item(), d1_valid.max().item()) if n_valid > 0 else (0, 0)
    debug_info["d2_range"] = (d2_valid.min().item(), d2_valid.max().item()) if n_valid > 0 else (0, 0)

    if n_valid < 10:
        return 1.0, 0.0, n_valid, debug_info

    # Compute per-pixel scale ratios
    ratios = d1_valid / d2_valid

    # Debug: ratio statistics
    debug_info["ratio_range"] = (ratios.min().item(), ratios.max().item())
    debug_info["ratio_median"] = ratios.median().item()
    debug_info["ratio_std"] = ratios.std().item()

    # Use median ratio (robust to outliers)
    scale = ratios.median().item()

    # Constraint: keep scale in reasonable range [0.5, 2.0]
    # If scale is outside this range, there's likely a problem
    original_scale = scale
    scale = max(0.5, min(2.0, scale))
    debug_info["scale_clamped"] = (original_scale != scale)
    debug_info["original_scale"] = original_scale

    # Count "inliers" - pixels where aligned depth is within 10% of reference
    aligned = scale * d2_valid
    relative_error = torch.abs(d1_valid - aligned) / (d1_valid + 1e-6)
    inliers = (relative_error < 0.1).sum().item()
    debug_info["inlier_ratio"] = inliers / n_valid if n_valid > 0 else 0

    return scale, 0.0, n_valid, debug_info  # shift is always 0!


def build_adjacency_graph(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    min_overlap: float = 0.1,
) -> dict[int, list[int]]:
    """Build a graph of which views overlap with each other.

    Args:
        extrinsics: [N, 4, 4] batch of extrinsics
        intrinsics: [4, 4] shared intrinsics
        min_overlap: Minimum overlap fraction to consider views adjacent

    Returns:
        Dictionary mapping view index to list of adjacent view indices
    """
    num_views = extrinsics.shape[0]
    f_px = intrinsics[0, 0].item()
    cx = intrinsics[0, 2].item()
    depth_size = int(cx * 2)  # Derive from principal point (cx = width/2)
    half_fov = math.atan((depth_size / 2) / f_px)

    adjacency = {i: [] for i in range(num_views)}

    for i in range(num_views):
        for j in range(i + 1, num_views):
            overlap = compute_view_overlap(extrinsics[i], extrinsics[j], half_fov)
            if overlap >= min_overlap:
                adjacency[i].append(j)
                adjacency[j].append(i)

    return adjacency


def compute_global_alignments(
    depth_maps: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    reference_view: int,
    num_iterations: int,
    inlier_threshold: float,
) -> dict[int, tuple[float, float]]:
    """Compute alignment parameters for all views relative to reference.

    Uses BFS to chain pairwise alignments through the view graph.
    DEPRECATED: Use compute_global_alignments_optimized instead.

    Args:
        depth_maps: [N, H, W, C] depth maps
        extrinsics: [N, 4, 4] extrinsics
        intrinsics: [4, 4] intrinsics
        reference_view: Index of reference view
        num_iterations: RANSAC iterations
        inlier_threshold: RANSAC inlier threshold

    Returns:
        Dictionary mapping view index to (scale, shift) tuple
    """
    num_views = depth_maps.shape[0]
    depth_h, depth_w = depth_maps.shape[1:3]
    device = depth_maps.device

    # Build adjacency graph
    adjacency = build_adjacency_graph(extrinsics, intrinsics)

    # BFS from reference view
    alignments = {reference_view: (1.0, 0.0)}  # Reference is unchanged
    visited = {reference_view}
    queue = deque([reference_view])

    pairwise_results = {}  # Cache pairwise alignments

    while queue:
        current = queue.popleft()
        current_scale, current_shift = alignments[current]

        for neighbor in adjacency[current]:
            if neighbor in visited:
                continue

            # Compute overlap mask and coordinates
            mask_c, mask_n, coords_c, coords_n = compute_overlap_mask(
                depth_h, depth_w,
                extrinsics[current], extrinsics[neighbor],
                intrinsics, device
            )

            if coords_c.shape[0] < 10:
                # Not enough overlap, skip this edge
                continue

            # Get depth maps (single channel)
            depth_c = depth_maps[current, :, :, 0]
            depth_n = depth_maps[neighbor, :, :, 0]

            # Apply current view's alignment to get aligned reference depth
            aligned_depth_c = current_scale * depth_c + current_shift

            # Find alignment from neighbor to aligned current
            # aligned_depth_c = scale * depth_n (NO SHIFT!)
            scale, shift, n_valid, debug_info = median_scale_alignment(
                aligned_depth_c, depth_n,
                coords_c, coords_n,
                debug=True
            )

            # Detailed debug output
            log.debug(f"View {neighbor} -> {current}:")
            log.debug(f"    Overlap: {debug_info['overlap_pixels']} pixels, {debug_info['valid_pixels']} valid")
            log.debug(f"    Depth1 (ref): {debug_info['d1_range'][0]:.2f} - {debug_info['d1_range'][1]:.2f}")
            log.debug(f"    Depth2 (src): {debug_info['d2_range'][0]:.2f} - {debug_info['d2_range'][1]:.2f}")
            log.debug(f"    Ratio range: {debug_info['ratio_range'][0]:.3f} - {debug_info['ratio_range'][1]:.3f}")
            log.debug(f"    Scale: {scale:.4f} (median={debug_info['ratio_median']:.4f}, std={debug_info['ratio_std']:.4f})")
            if debug_info.get('scale_clamped'):
                log.warning(f"Scale clamped from {debug_info['original_scale']:.4f} to {scale:.4f}")
            log.debug(f"    Inlier ratio: {debug_info['inlier_ratio']:.1%}")

            alignments[neighbor] = (scale, shift)
            visited.add(neighbor)
            queue.append(neighbor)

    # Handle disconnected views (no overlap with any aligned view)
    for i in range(num_views):
        if i not in alignments:
            log.warning(f"View {i} has no overlap with aligned views, using identity")
            alignments[i] = (1.0, 0.0)

    return alignments


def compute_global_misalignment(
    scales: list[float],
    pairwise_ratios: dict[tuple[int, int], float],
) -> tuple[float, float, dict]:
    """Compute global misalignment loss given scales and measured ratios.

    The loss measures how well the solved scales satisfy the pairwise constraints.

    Args:
        scales: List of scale values [s_0, s_1, ..., s_N]
        pairwise_ratios: Dict mapping (i,j) to measured ratio d_i/d_j

    Returns:
        Tuple of (total_loss, mean_loss, per_pair_losses)
    """
    per_pair_losses = {}
    total_loss = 0.0

    for (i, j), measured_ratio in pairwise_ratios.items():
        # We want: s_i * d_i = s_j * d_j in overlap
        # So: s_i / s_j should equal d_j / d_i = 1/measured_ratio
        target_ratio = 1.0 / measured_ratio
        actual_ratio = scales[i] / scales[j] if scales[j] > 0 else float('inf')

        # Loss in log-space (scale-invariant)
        if actual_ratio > 0 and target_ratio > 0:
            log_error = abs(math.log(actual_ratio) - math.log(target_ratio))
            per_pair_losses[(i, j)] = log_error
            total_loss += log_error ** 2
        else:
            per_pair_losses[(i, j)] = float('inf')

    num_pairs = len(pairwise_ratios)
    mean_loss = math.sqrt(total_loss / num_pairs) if num_pairs > 0 else 0.0

    return total_loss, mean_loss, per_pair_losses


def compute_global_alignments_optimized(
    depth_maps: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> dict[int, tuple[float, float]]:
    """Compute globally optimal scale alignments using least squares.

    Instead of BFS chaining (which causes cumulative drift), solve for all
    scales jointly by minimizing total pairwise inconsistency.

    The formulation:
    - For each pair (i,j) we measure ratio r_ij = median(d_i / d_j)
    - We want scales s_i such that s_i / s_j ~= r_ij
    - In log-space: x_i - x_j = log(r_ij) where x_i = log(s_i)
    - This is a linear system A @ x = b, solved with least squares

    Args:
        depth_maps: [N, H, W, C] depth maps
        extrinsics: [N, 4, 4] extrinsics
        intrinsics: [4, 4] intrinsics

    Returns:
        Dictionary mapping view index to (scale, shift=0.0) tuple
    """
    num_views = depth_maps.shape[0]
    depth_h, depth_w = depth_maps.shape[1:3]
    device = depth_maps.device

    log.info(f"Computing globally optimal alignment for {num_views} views")

    # Step 1: Compute ALL pairwise scale ratios
    adjacency = build_adjacency_graph(extrinsics, intrinsics)
    pairwise_ratios = {}  # (i, j) -> ratio where i < j

    log.info("Computing pairwise ratios...")
    for i in range(num_views):
        for j in adjacency[i]:
            if i < j:  # Only compute once per pair
                # Compute overlap
                mask_i, mask_j, coords_i, coords_j = compute_overlap_mask(
                    depth_h, depth_w, extrinsics[i], extrinsics[j],
                    intrinsics, device
                )

                if coords_i.shape[0] < 10:
                    log.debug(f"Pair ({i},{j}): insufficient overlap ({coords_i.shape[0]} pixels)")
                    continue

                depth_i = depth_maps[i, :, :, 0]
                depth_j = depth_maps[j, :, :, 0]

                # Get median ratio: d_i / d_j
                scale, _, n_valid, debug_info = median_scale_alignment(
                    depth_i, depth_j, coords_i, coords_j, debug=True
                )

                if n_valid > 100:
                    pairwise_ratios[(i, j)] = scale
                    log.debug(f"Pair ({i},{j}): ratio={scale:.4f}, "
                             f"valid={n_valid}, inlier_ratio={debug_info['inlier_ratio']:.1%}")
                else:
                    log.debug(f"Pair ({i},{j}): insufficient valid pixels ({n_valid})")

    # Compute initial loss (with identity scales)
    identity_scales = [1.0] * num_views
    initial_loss, initial_rmse, _ = compute_global_misalignment(identity_scales, pairwise_ratios)
    log.info(f"Initial misalignment (identity scales): loss={initial_loss:.4f}, RMSE={initial_rmse:.4f}")

    # Step 2: Build linear system for log-scale optimization
    # x_i - x_j = log(r_ij) for each pair
    # Fix x_0 = 0 (reference)

    num_pairs = len(pairwise_ratios)
    log.info(f"Found {num_pairs} valid pairs")

    if num_pairs == 0:
        log.warning("No valid pairs, using identity scales")
        return {i: (1.0, 0.0) for i in range(num_views)}

    # Build A matrix and b vector
    # For N views with reference fixed, we have N-1 unknowns
    # A is [num_pairs, num_views-1], b is [num_pairs]
    A = torch.zeros(num_pairs, num_views - 1, device=device)
    b = torch.zeros(num_pairs, device=device)

    for k, ((i, j), ratio) in enumerate(pairwise_ratios.items()):
        # Equation: x_i - x_j = log(ratio)
        # ratio = d_i / d_j, so we want s_i * d_i = s_j * d_j
        # which means s_i / s_j = d_j / d_i = 1/ratio
        # So: x_i - x_j = -log(ratio) = log(1/ratio)

        # Actually, let's think about this more carefully:
        # We measured: ratio = median(d_i / d_j)
        # After alignment: s_i * d_i should equal s_j * d_j in overlap
        # So: s_i / s_j = d_j / d_i = 1/ratio
        # In log: x_i - x_j = log(1/ratio) = -log(ratio)

        log_ratio = -math.log(max(ratio, 1e-6))  # s_i / s_j = 1/ratio

        if i == 0:
            # x_0 = 0, so: -x_j = log_ratio -> x_j = -log_ratio
            A[k, j - 1] = -1.0
        elif j == 0:
            # x_0 = 0, so: x_i = log_ratio
            A[k, i - 1] = 1.0
        else:
            # x_i - x_j = log_ratio
            A[k, i - 1] = 1.0
            A[k, j - 1] = -1.0

        b[k] = log_ratio

    # Step 3: Solve least squares: x = (A^T A)^{-1} A^T b
    log.info(f"Solving least squares system ({num_pairs} equations, {num_views-1} unknowns)")

    try:
        # Use torch.linalg.lstsq for numerical stability
        solution = torch.linalg.lstsq(A, b).solution  # [num_views-1]

        # Convert back from log-space
        scales = [1.0]  # x_0 = 0 -> s_0 = 1
        for i in range(num_views - 1):
            scales.append(math.exp(solution[i].item()))

        log.debug(f"Raw scales: {[f'{s:.4f}' for s in scales]}")

        # Compute loss after optimization (before normalization)
        opt_loss, opt_rmse, per_pair = compute_global_misalignment(scales, pairwise_ratios)
        log.info(f"Optimized misalignment: loss={opt_loss:.4f}, RMSE={opt_rmse:.4f}")

    except Exception as e:
        log.error(f"Least squares failed: {e}, using identity scales")
        scales = [1.0] * num_views

    # Step 4: Normalize so median scale = 1.0 (prevents overall drift)
    median_scale = sorted(scales)[len(scales) // 2]
    if median_scale > 0:
        scales = [s / median_scale for s in scales]

    log.debug(f"Normalized scales: {[f'{s:.4f}' for s in scales]}")

    # Compute final loss after normalization
    final_loss, final_rmse, per_pair = compute_global_misalignment(scales, pairwise_ratios)
    log.info(f"Final misalignment: loss={final_loss:.4f}, RMSE={final_rmse:.4f}")

    # Print per-pair residuals
    log.debug("Per-pair log-errors:")
    for (i, j), err in sorted(per_pair.items()):
        status = "OK" if err < 0.1 else "WARN" if err < 0.2 else "BAD"
        log.debug(f"    Pair ({i},{j}): {err:.4f} [{status}]")

    # Summary
    min_scale, max_scale = min(scales), max(scales)
    variation = max_scale / min_scale if min_scale > 0 else float('inf')
    log.info(f"Scale range: {min_scale:.4f} - {max_scale:.4f} ({variation:.2f}x variation)")

    improvement = (initial_rmse - final_rmse) / initial_rmse * 100 if initial_rmse > 0 else 0
    log.info(f"Improvement: {improvement:.1f}% reduction in RMSE")

    # Warn if variation is still high
    if variation > 2.0:
        log.warning(f"High scale variation ({variation:.2f}x) - alignment may be unreliable")

    return {i: (scales[i], 0.0) for i in range(num_views)}


class AlignDepthMaps(io.ComfyNode):
    """Global optimization-based scale alignment for multi-view depth maps.

    Aligns depth maps to remove seams caused by scale ambiguity in
    monocular depth estimation. Uses least-squares optimization in log-space
    to find globally consistent scales.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AlignDepthMaps",
            display_name="Align Depth Maps (Global Optimization)",
            category="SHARP",
            description="Align depth maps using global optimization to remove seams from monocular depth estimation.",
            inputs=[
                io.Image.Input("depth_maps"),
                io.Custom("EXTRINSICS").Input("extrinsics"),
                io.Custom("INTRINSICS").Input("intrinsics"),
                io.Combo.Input("method",
                               options=["global_optimization", "bfs_chain"],
                               default="global_optimization", optional=True,
                               tooltip="global_optimization: solve all scales jointly (recommended). bfs_chain: chain pairwise alignments (can drift)."),
                io.Int.Input("reference_view", default=0, min=0, max=999, step=1,
                             optional=True,
                             tooltip="Index of the reference view (scale=1.0)"),
            ],
            outputs=[
                io.Image.Output(display_name="aligned_depth_maps"),
                io.Custom("EXTRINSICS").Output(display_name="extrinsics"),
                io.Custom("INTRINSICS").Output(display_name="intrinsics"),
                io.String.Output(display_name="alignment_info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        depth_maps: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        method: str = "global_optimization",
        reference_view: int = 0,
    ):
        """Align depth maps using global optimization or BFS chain."""

        num_views = depth_maps.shape[0]
        log.info(f"Aligning {num_views} depth maps")
        log.info(f"Method: {method}")

        # Clamp reference view to valid range
        reference_view = min(reference_view, num_views - 1)

        # Compute alignments using selected method
        if method == "global_optimization":
            alignments = compute_global_alignments_optimized(
                depth_maps, extrinsics, intrinsics
            )
        else:
            # Legacy BFS method (kept for comparison)
            log.info(f"Using BFS chain (reference view: {reference_view})")
            alignments = compute_global_alignments(
                depth_maps, extrinsics, intrinsics,
                reference_view, num_iterations=1000, inlier_threshold=0.05
            )

        # Apply alignments
        aligned_depth_maps = []
        alignment_info = {"method": method}

        for i in range(num_views):
            scale, shift = alignments[i]
            depth = depth_maps[i]  # [H, W, C]

            # Apply alignment (scale only, shift is always 0)
            aligned = scale * depth + shift

            aligned_depth_maps.append(aligned)
            alignment_info[f"view_{i}"] = {"scale": scale, "shift": shift}

        # Stack result
        aligned_batch = torch.stack(aligned_depth_maps, dim=0)

        # Convert alignment info to JSON string
        info_str = json.dumps(alignment_info, indent=2)

        scales = [alignments[i][0] for i in range(num_views)]
        min_scale, max_scale = min(scales), max(scales)
        variation = max_scale / min_scale if min_scale > 0 else float('inf')

        log.info("Alignment complete")
        log.info(f"Final scale range: {min_scale:.4f} - {max_scale:.4f} ({variation:.2f}x variation)")

        # Check for problematic alignments
        if variation > 2.0:
            log.warning(f"Large scale variation ({variation:.2f}x) - alignment may be unreliable.")

        return io.NodeOutput(aligned_batch, extrinsics, intrinsics, info_str)


NODE_CLASS_MAPPINGS = {
    "AlignDepthMaps": AlignDepthMaps,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlignDepthMaps": "Align Depth Maps (Global Optimization)",
}
