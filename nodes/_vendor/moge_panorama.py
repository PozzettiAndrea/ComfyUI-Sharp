import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
from typing import *
import itertools
import json
import warnings

import cv2
import numpy as np
import torch
from numpy import ndarray
from tqdm import tqdm, trange
from scipy.sparse import csr_array, hstack, vstack
from scipy.ndimage import convolve
from scipy.sparse.linalg import lsmr

import utils3d


def get_panorama_cameras():
    vertices, _ = utils3d.np.create_icosahedron_mesh()
    intrinsics = utils3d.np.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics = utils3d.np.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return extrinsics, [intrinsics] * len(vertices)


def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return directions


def directions_to_spherical_uv(directions: np.ndarray):
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    u = 1 - np.arctan2(directions[..., 1], directions[..., 0]) / (2 * np.pi) % 1.0
    v = np.arccos(directions[..., 2]) / np.pi
    return np.stack([u, v], axis=-1)


def split_panorama_image(image: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, resolution: int):
    height, width = image.shape[:2]
    uv = utils3d.np.uv_map((resolution, resolution))
    splitted_images = []
    for i in range(len(extrinsics)):
        spherical_uv = directions_to_spherical_uv(utils3d.np.unproject_cv(uv, np.ones_like(uv[..., 0]), extrinsics=extrinsics[i], intrinsics=intrinsics[i]))
        pixels = utils3d.np.uv_to_pixel(spherical_uv, (height, width)).astype(np.float32)

        splitted_image = cv2.remap(image, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR)
        splitted_images.append(splitted_image)
    return splitted_images


def split_panorama_image_gpu(image: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, resolution: int):
    """Batched-GPU equivalent of `split_panorama_image`.

    Same math: for each of N face cameras, build a (resolution, resolution)
    lookup of spherical UVs from the panorama, and bilinearly resample.
    Difference: the lookup is computed once for all N faces as a batched
    (N, R, R, 2) grid (np vectorized over the face dim) and the per-face
    resampling collapses to one F.grid_sample call. cv2.remap is single-
    threaded per call and was the dominant cost on N=42 face splits.

    Bilinear matches cv2.INTER_LINEAR to within fp32 round-off when we use
    `align_corners=False` and convert the cv2 pixel coordinate `p` (where
    `p=0` is the centre of pixel 0) to the grid_sample coordinate
    `(2p + 1) / W - 1`. Panoramic wrap in U is handled by the
    `(% 1.0)` inside `directions_to_spherical_uv` so the lookups never
    exit the image in U; in V we clip to [0, 1] which is what cv2 does
    by default (BORDER_CONSTANT 0 below the south pole, but those samples
    are unreachable because the icosahedron forwards are constrained to
    the sphere).

    Falls back to the CPU implementation if CUDA isn't available.
    """
    if not torch.cuda.is_available():
        return split_panorama_image(image, extrinsics, intrinsics, resolution)

    height, width = image.shape[:2]
    N = len(extrinsics)
    device = torch.device("cuda")

    # Build the (R, R, 2) UV map once on CPU (cheap, sub-ms).
    uv = utils3d.np.uv_map((resolution, resolution))

    # Per-face lookup pixels in panorama pixel coordinates (cv2 convention:
    # pixel centre = integer). Stack into (N, R, R, 2) before going to GPU.
    pixel_lookups = np.empty((N, resolution, resolution, 2), dtype=np.float32)
    for i in range(N):
        spherical_uv = directions_to_spherical_uv(
            utils3d.np.unproject_cv(
                uv, np.ones_like(uv[..., 0]),
                extrinsics=extrinsics[i], intrinsics=intrinsics[i],
            )
        )
        pixel_lookups[i] = utils3d.np.uv_to_pixel(
            spherical_uv, (height, width)
        ).astype(np.float32)

    # Convert pixel coords -> grid_sample normalized coords. Use the
    # half-pixel-centre convention so this lines up with cv2.remap's
    # INTER_LINEAR.
    grid = np.empty_like(pixel_lookups)
    grid[..., 0] = (2.0 * pixel_lookups[..., 0] + 1.0) / float(width) - 1.0
    grid[..., 1] = (2.0 * pixel_lookups[..., 1] + 1.0) / float(height) - 1.0

    grid_t = torch.from_numpy(grid).to(device)  # (N, R, R, 2) fp32

    # Image to (1, C, H, W) -> broadcast to (N, C, H, W) for batched sample.
    img_chw = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    if img_chw.ndim == 2:
        img_chw = img_chw.unsqueeze(-1)   # (H, W) -> (H, W, 1)
    img_chw = img_chw.permute(2, 0, 1).contiguous().to(torch.float32)  # (C, H, W)
    img_batched = img_chw.unsqueeze(0).expand(N, -1, -1, -1)            # (N, C, H, W)

    sampled = torch.nn.functional.grid_sample(
        img_batched, grid_t,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )  # (N, C, R, R)

    sampled = sampled.permute(0, 2, 3, 1).contiguous()  # (N, R, R, C)
    arr = sampled.cpu().numpy()
    if image.dtype == np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    elif image.dtype == np.float32:
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(image.dtype)
    # Drop the C dim if input was 2-D (single-channel mask etc.)
    if image.ndim == 2:
        arr = arr[..., 0]
    return [arr[i] for i in range(N)]


def poisson_equation(width: int, height: int, wrap_x: bool = False, wrap_y: bool = False) -> Tuple[csr_array, ndarray]:
    grid_index = np.arange(height * width).reshape(height, width)
    grid_index = np.pad(grid_index, ((0, 0), (1, 1)), mode='wrap' if wrap_x else 'edge')
    grid_index = np.pad(grid_index, ((1, 1), (0, 0)), mode='wrap' if wrap_y else 'edge')
    
    data = np.array([[-4, 1, 1, 1, 1]], dtype=np.float32).repeat(height * width, axis=0).reshape(-1)
    indices = np.stack([
        grid_index[1:-1, 1:-1],
        grid_index[:-2, 1:-1],         # up
        grid_index[2:, 1:-1],          # down
        grid_index[1:-1, :-2],         # left
        grid_index[1:-1, 2:]           # right
    ], axis=-1).reshape(-1)                                                                 
    indptr = np.arange(0, height * width * 5 + 1, 5) 
    A = csr_array((data, indices, indptr), shape=(height * width, height * width))
    
    return A


def grad_equation(width: int, height: int, wrap_x: bool = False, wrap_y: bool = False) -> Tuple[csr_array, np.ndarray]:
    grid_index = np.arange(width * height).reshape(height, width)
    if wrap_x:
        grid_index = np.pad(grid_index, ((0, 0), (0, 1)), mode='wrap')
    if wrap_y:
        grid_index = np.pad(grid_index, ((0, 1), (0, 0)), mode='wrap')

    data = np.concatenate([
        np.concatenate([
            np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),        # x[i,j]                                           
            -np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),       # x[i,j-1]           
        ], axis=1).reshape(-1),
        np.concatenate([
            np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),        # x[i,j]                                           
            -np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),       # x[i-1,j]           
        ], axis=1).reshape(-1),
    ])
    indices = np.concatenate([
        np.concatenate([
            grid_index[:, :-1].reshape(-1, 1),
            grid_index[:, 1:].reshape(-1, 1),
        ], axis=1).reshape(-1),
        np.concatenate([
            grid_index[:-1, :].reshape(-1, 1),
            grid_index[1:, :].reshape(-1, 1),
        ], axis=1).reshape(-1),
    ])
    indptr = np.arange(0, grid_index.shape[0] * (grid_index.shape[1] - 1) * 2 + (grid_index.shape[0] - 1) * grid_index.shape[1] * 2 + 1, 2)
    A = csr_array((data, indices, indptr), shape=(grid_index.shape[0] * (grid_index.shape[1] - 1) + (grid_index.shape[0] - 1) * grid_index.shape[1], height * width))

    return A


def merge_panorama_depth(width: int, height: int, distance_maps: List[np.ndarray], pred_masks: List[np.ndarray], extrinsics: List[np.ndarray], intrinsics: List[np.ndarray], *, pbar=None, _top_size=None):
    """Recursive multi-resolution merge of N rectilinear face depth maps -> equirect depth.

    Optional `pbar` is a ComfyUI ProgressBar-like object (any obj with
    `update_absolute(value, total)`). Progress is weighted by `pixels^1.5`
    per level to match LSMR's asymptotic cost — the top level dominates,
    so the bar advances slowly until the unwind reaches it.
    """
    import sys
    import time

    def _p(msg):
        print(f"[merge_panorama_depth {width:>4}x{height:<4}] {msg}", file=sys.stderr, flush=True)

    # --- Progress-bar setup (one-time on top-level call) ---------------------
    if _top_size is None:
        _top_size = (width, height)
    if pbar is not None and not hasattr(pbar, "_wn_levels"):
        levels = []
        w_, h_ = _top_size
        while True:
            levels.append((w_, h_))
            if max(w_, h_) <= 256:
                break
            w_, h_ = w_ // 2, h_ // 2
        levels.reverse()  # smallest -> largest, matches recursion unwind order
        weights = np.array([(lw * lh) ** 1.5 for lw, lh in levels], dtype=np.float64)
        weights = weights / weights.sum()
        pbar._wn_levels = levels
        pbar._wn_weights = weights
        _p(f"progress bar: {len(levels)} levels — weights {[f'{x:.1%}' for x in weights]}")

    def _bump(level_frac: float):
        """Advance the global progress bar to (cum % of prior levels) + (level_frac × this level's share)."""
        if pbar is None or not hasattr(pbar, "_wn_levels"):
            return
        try:
            i = pbar._wn_levels.index((width, height))
        except ValueError:
            return
        pct_floor = float(pbar._wn_weights[:i].sum())
        pct_ceil  = float(pbar._wn_weights[:i + 1].sum())
        pct = pct_floor + (pct_ceil - pct_floor) * max(0.0, min(1.0, level_frac))
        pbar.update_absolute(int(min(99, pct * 100)), 100)

    t_lvl_start = time.perf_counter()
    _bump(0.0)

    if max(width, height) > 256:
        _p(f"recurse into {width // 2}x{height // 2}…")
        panorama_depth_init, _ = merge_panorama_depth(
            width // 2, height // 2, distance_maps, pred_masks, extrinsics, intrinsics,
            pbar=pbar, _top_size=_top_size,
        )
        _p(f"recursion returned; upsampling {width // 2}x{height // 2} -> {width}x{height}")
        panorama_depth_init = cv2.resize(panorama_depth_init, (width, height), cv2.INTER_LINEAR)
    else:
        _p(f"base case (<=256 on each side) -> no recursion, no init x0 for LSMR")
        panorama_depth_init = None
    _bump(0.05)

    _p(f"building sphere uv + ray directions ({width}×{height})…")
    uv = utils3d.np.uv_map(height, width)
    spherical_directions = spherical_uv_to_directions(uv)

    # Warp each view to the panorama
    _p(f"warping {len(distance_maps)} face depth maps onto the equirect…")
    t_warp_start = time.perf_counter()
    panorama_log_distance_grad_maps, panorama_grad_masks = [], []
    panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
    panorama_pred_masks = []
    for i in range(len(distance_maps)):
        projected_uv, projected_depth = utils3d.np.project_cv(spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        projection_valid_mask = (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)
        
        projected_pixels = utils3d.np.uv_to_pixel(np.clip(projected_uv, 0, 1), distance_maps[i].shape).astype(np.float32)
        
        log_splitted_distance = np.log(distance_maps[i])
        panorama_log_distance_map = np.where(projection_valid_mask, cv2.remap(log_splitted_distance, projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
        panorama_pred_mask = projection_valid_mask & (cv2.remap(pred_masks[i].astype(np.uint8), projected_pixels[..., 0], projected_pixels[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0)

        # calculate gradient map
        padded = np.pad(panorama_log_distance_map, ((0, 0), (0, 1)), mode='wrap')
        grad_x, grad_y = padded[:, :-1] - padded[:, 1:], padded[:-1, :] - padded[1:, :]

        padded = np.pad(panorama_pred_mask, ((0, 0), (0, 1)), mode='wrap')
        mask_x, mask_y = padded[:, :-1] & padded[:, 1:], padded[:-1, :] & padded[1:, :]
        
        panorama_log_distance_grad_maps.append((grad_x, grad_y))
        panorama_grad_masks.append((mask_x, mask_y))

        # calculate laplacian map
        padded = np.pad(panorama_log_distance_map, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        laplacian = convolve(padded, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32))[1:-1, 1:-1]

        padded = np.pad(panorama_pred_mask, ((1, 1), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (1, 1)), mode='wrap')
        mask = convolve(padded.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))[1:-1, 1:-1] == 5

        panorama_log_distance_laplacian_maps.append(laplacian)
        panorama_laplacian_masks.append(mask)
        
        panorama_pred_masks.append(panorama_pred_mask)
        if (i + 1) % 10 == 0 or i == len(distance_maps) - 1:
            _p(f"  warped {i + 1}/{len(distance_maps)} faces")

    _bump(0.45)
    _p(f"warp stage done in {time.perf_counter() - t_warp_start:.2f}s; "
       f"stacking + averaging gradient/laplacian maps…")
    panorama_log_distance_grad_x = np.stack([grad_map[0] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_log_distance_grad_y = np.stack([grad_map[1] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_grad_mask_x = np.stack([mask_map[0] for mask_map in panorama_grad_masks], axis=0)
    panorama_grad_mask_y = np.stack([mask_map[1] for mask_map in panorama_grad_masks], axis=0)

    panorama_log_distance_grad_x = np.sum(panorama_log_distance_grad_x * panorama_grad_mask_x, axis=0) / np.sum(panorama_grad_mask_x, axis=0).clip(1e-3)
    panorama_log_distance_grad_y = np.sum(panorama_log_distance_grad_y * panorama_grad_mask_y, axis=0) / np.sum(panorama_grad_mask_y, axis=0).clip(1e-3)

    panorama_laplacian_maps = np.stack(panorama_log_distance_laplacian_maps, axis=0)
    panorama_laplacian_masks = np.stack(panorama_laplacian_masks, axis=0)
    panorama_laplacian_map = np.sum(panorama_laplacian_maps * panorama_laplacian_masks, axis=0) / np.sum(panorama_laplacian_masks, axis=0).clip(1e-3)

    grad_x_mask = np.any(panorama_grad_mask_x, axis=0).reshape(-1)
    grad_y_mask = np.any(panorama_grad_mask_y, axis=0).reshape(-1)
    grad_mask = np.concatenate([grad_x_mask, grad_y_mask])
    laplacian_mask = np.any(panorama_laplacian_masks, axis=0).reshape(-1)

    _bump(0.55)
    # Solve overdetermined system
    _p(f"building sparse linear system (grad_equation + poisson_equation)…")
    t_build = time.perf_counter()
    A = vstack([
        grad_equation(width, height, wrap_x=True, wrap_y=False)[grad_mask],
        poisson_equation(width, height, wrap_x=True, wrap_y=False)[laplacian_mask],
    ])
    b = np.concatenate([
        panorama_log_distance_grad_x.reshape(-1)[grad_x_mask],
        panorama_log_distance_grad_y.reshape(-1)[grad_y_mask],
        panorama_laplacian_map.reshape(-1)[laplacian_mask]
    ])
    _p(f"linear system: A {A.shape} ({A.nnz} nnz), b {b.shape} — built in "
       f"{time.perf_counter() - t_build:.2f}s")
    _bump(0.65)

    _p(f"LSMR solve (atol=1e-5, btol=1e-5, "
       f"x0={'provided' if panorama_depth_init is not None else 'None'})…")
    t_solve = time.perf_counter()
    x, istop, itn, normr, normar, norma, conda, normx = lsmr(
        A, b,
        atol=1e-5, btol=1e-5,
        x0=np.log(panorama_depth_init).reshape(-1) if panorama_depth_init is not None else None,
        show=False,
    )
    _p(f"LSMR done in {time.perf_counter() - t_solve:.2f}s: "
       f"{itn} iterations, residual={normr:.4f}, istop={istop}")
    _bump(0.95)

    panorama_depth = np.exp(x).reshape(height, width).astype(np.float32)
    panorama_mask = np.any(panorama_pred_masks, axis=0)

    _p(f"level done in {time.perf_counter() - t_lvl_start:.2f}s; "
       f"depth: min={float(panorama_depth.min()):.3f}, "
       f"median={float(np.median(panorama_depth)):.3f}, "
       f"max={float(panorama_depth.max()):.3f}; "
       f"valid pixels: {int(panorama_mask.sum())}/{panorama_mask.size}")
    _bump(1.0)
    return panorama_depth, panorama_mask
         
