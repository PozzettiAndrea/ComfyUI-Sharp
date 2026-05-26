"""SharpRefinedDepthFromGaussians node for ComfyUI-Sharp.

Recovers a per-face "refined depth map" from the gaussian PLYs produced by
SharpPredict, by reprojecting each face's layer-0 (visible-surface)
gaussians back into that face's camera frustum.

Why: SHARP's `SharpPredictDepth` exposes the pre-refinement disparity head
output. The full gaussian decoder produces a refined per-gaussian z that's
typically tighter than the disparity head's prediction. Those refined z's
ARE in the saved PLY (in world coords after `unproject_gaussians`); this
node inverts that unproject to recover the camera-space depth they
represent.

Pipeline shape:
    SamplePanorama / SharpPanoramaCubeSplit
        -> images, ext, intr
    SharpPredict (batch)
        -> ply_path (folder with N PLYs), ext, intr
    SharpRefinedDepthFromGaussians (this node)
        -> depth_maps [N, H, W, 3]    # refined per-face depth
    AlignDepthMaps
        -> globally scaled depth_maps
    (multiply scales back into the gaussians + MergeGaussians)

Algorithm per face i:
    g_world = layer-0 gaussian positions in world coords (loaded from PLY)
    cam = R_w2c[i] @ g_world + t_w2c[i]              # world -> camera
    z   = cam[:, 2]                                  # depth along view axis
    valid = z > 0                                     # in front of camera
    u = fx[i] * cam[:, 0] / z + cx[i]                 # face pixel coords
    v = fy[i] * cam[:, 1] / z + cy[i]
    depth_map[v, u] = min(depth_map[v, u], z)         # nearest gaussian per pixel
                                                       # = layer-0 by construction
                                                       # (layer-1 is behind layer-0)

Output is sparse — only pixels where a layer-0 gaussian landed get a depth.
Other pixels are 0 + masked invalid.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch

from comfy_api.latest import io

log = logging.getLogger("sharp")


def _load_gaussian_positions(ply_path: Path) -> np.ndarray:
    """Read just the x/y/z fields from a Sharp-format PLY. Returns [N, 3]."""
    from plyfile import PlyData
    plydata = PlyData.read(str(ply_path))
    vertex = plydata['vertex']
    return np.stack([
        np.asarray(vertex['x']),
        np.asarray(vertex['y']),
        np.asarray(vertex['z']),
    ], axis=-1).astype(np.float32)


def _list_ply_paths(ply_path_str: str, expected_n: int) -> list[Path]:
    """Resolve the ply_path input (SharpPredict output) to a list of PLYs.

    SharpPredict writes:
      - single image: <prefix>_<ts>.ply
      - batch:        <prefix>_<ts>/001.ply, 002.ply, ...
    """
    p = Path(ply_path_str)
    if p.is_file() and p.suffix.lower() == ".ply":
        if expected_n != 1:
            raise ValueError(
                f"ply_path is a single file ({p.name}) but extrinsics has "
                f"N={expected_n}. Pass the SharpPredict batch folder instead."
            )
        return [p]
    if p.is_dir():
        plies = sorted(
            child for child in p.iterdir()
            if child.is_file() and child.suffix.lower() == ".ply"
        )
        if len(plies) != expected_n:
            raise ValueError(
                f"ply_path directory {p} has {len(plies)} PLYs but extrinsics "
                f"has N={expected_n}. Did SharpPredict run on the same batch?"
            )
        return plies
    raise FileNotFoundError(f"ply_path {ply_path_str!r} is not a file or directory")


def _project_gaussians_to_depth(
    gauss_world: torch.Tensor,
    ext_w2c: torch.Tensor,
    K_pixel: torch.Tensor,
    out_h: int,
    out_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project N gaussian positions into one face camera, return depth map.

    Args:
        gauss_world: [G, 3] world-space gaussian positions
        ext_w2c:     [4, 4] world-to-camera matrix for this face
        K_pixel:     [3, 3] pixel-space intrinsics
        out_h, out_w: output depth map size

    Returns:
        depth_map: [H, W] float; depth at each pixel where a gaussian landed,
                   0 elsewhere. Per-pixel = min over all gaussians that hit
                   that pixel (so layer-0 is taken automatically since
                   layer-1 sits behind layer-0).
        valid:     [H, W] bool; True where at least one gaussian landed.
    """
    device = gauss_world.device

    # world -> camera (homogeneous)
    R = ext_w2c[:3, :3]
    t = ext_w2c[:3, 3]
    cam = gauss_world @ R.T + t  # [G, 3]
    z = cam[:, 2]

    # Project to pixel coords. Only consider gaussians in front of camera.
    fx = K_pixel[0, 0]
    fy = K_pixel[1, 1]
    cx = K_pixel[0, 2]
    cy = K_pixel[1, 2]

    in_front = z > 1e-6
    z_safe = torch.where(in_front, z, torch.ones_like(z))
    u = fx * cam[:, 0] / z_safe + cx
    v = fy * cam[:, 1] / z_safe + cy

    # Round to nearest pixel; require in-bounds.
    u_i = u.round().long()
    v_i = v.round().long()
    in_bounds = (u_i >= 0) & (u_i < out_w) & (v_i >= 0) & (v_i < out_h) & in_front

    # Scatter: keep MIN z per pixel (-> layer-0, visible surface).
    # 1-D flat pixel index for scatter_reduce.
    flat_idx = v_i * out_w + u_i
    # For invalid gaussians, point at index 0 with z = +inf so they lose every reduce.
    safe_idx = torch.where(in_bounds, flat_idx, torch.zeros_like(flat_idx))
    safe_z = torch.where(in_bounds, z, torch.full_like(z, float("inf")))

    depth_flat = torch.full((out_h * out_w,), float("inf"), dtype=torch.float32, device=device)
    depth_flat.scatter_reduce_(0, safe_idx, safe_z, reduce="amin", include_self=True)

    depth_map = depth_flat.view(out_h, out_w)
    valid = torch.isfinite(depth_map)
    depth_map = torch.where(valid, depth_map, torch.zeros_like(depth_map))

    return depth_map, valid


class SharpRefinedDepthFromGaussians(io.ComfyNode):
    """Per-face gaussian PLYs -> per-face refined depth maps (layer-0)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharpRefinedDepthFromGaussians",
            display_name="SHARP Refined Depth from Gaussians",
            category="SHARP",
            description=(
                "Reproject SharpPredict's PLY gaussians back into per-face "
                "camera space to recover the refined per-pixel depth (the "
                "actual z's the decoder produced, NOT the pre-refinement "
                "disparity head). Min-depth per pixel = layer 0 (visible "
                "surface).\n\n"
                "Feed depth_maps into AlignDepthMaps for log-scale "
                "alignment, then apply the resulting per-face scales back "
                "to the gaussians before MergeGaussians."
            ),
            inputs=[
                io.String.Input(
                    "ply_path",
                    tooltip="From SharpPredict.ply_path. Single .ply (for "
                            "batch_size=1) OR a folder with one .ply per "
                            "face (for batch). Folder file count must match "
                            "extrinsics N."),
                io.Custom("EXTRINSICS").Input(
                    "extrinsics",
                    tooltip="[N, 4, 4] world-to-camera per face. Same matrix "
                            "you passed to SharpPredict — typically from "
                            "SamplePanorama or SharpPanoramaCubeSplit."),
                io.Custom("INTRINSICS").Input(
                    "intrinsics",
                    tooltip="[N, 3, 3] (or [N, 4, 4]) pixel-space intrinsics "
                            "per face. The 3x3 K is used; any extra row/col "
                            "is ignored."),
                io.Int.Input(
                    "out_size", default=1536, min=256, max=2048, step=64,
                    tooltip="Output depth-map resolution per face (square). "
                            "1536 matches SHARP's internal resolution."),
            ],
            outputs=[
                io.Image.Output(
                    display_name="depth_maps",
                    tooltip="[N, H, W, 3] float depth maps. Depth value "
                            "broadcast across 3 channels so it composes "
                            "with regular IMAGE-consuming nodes. 0 where "
                            "no gaussian landed."),
                io.Mask.Output(
                    display_name="valid_mask",
                    tooltip="[N, H, W] float 0/1; 1 where a gaussian landed."),
                io.Custom("EXTRINSICS").Output(
                    display_name="extrinsics",
                    tooltip="Pass-through, so this node can sit upstream of "
                            "AlignDepthMaps without rewiring."),
                io.Custom("INTRINSICS").Output(
                    display_name="intrinsics",
                    tooltip="Pass-through."),
            ],
        )

    @classmethod
    def execute(
        cls, ply_path: str,
        extrinsics: torch.Tensor, intrinsics: torch.Tensor,
        out_size: int = 1536,
    ):
        # Normalize ext/intr shapes.
        ext = extrinsics.float()
        if ext.dim() == 2:
            ext = ext.unsqueeze(0)
        N = ext.shape[0]
        if ext.shape[1:] != (4, 4):
            raise ValueError(f"extrinsics must be [N, 4, 4], got {tuple(ext.shape)}")

        intr = intrinsics.float()
        if intr.dim() == 2:
            intr = intr.unsqueeze(0)
        if intr.shape[0] != N:
            raise ValueError(
                f"intrinsics N ({intr.shape[0]}) doesn't match extrinsics N ({N})"
            )
        # Accept 4x4 or 3x3.
        K_pixel = intr[:, :3, :3].contiguous()

        # Resolve to per-face PLY paths.
        ply_files = _list_ply_paths(ply_path, expected_n=N)
        log.info(
            f"[SharpRefinedDepthFromGaussians] loading {len(ply_files)} PLY(s) "
            f"-> projecting to {out_size}x{out_size} per face"
        )

        device = ext.device
        H_out = W_out = int(out_size)

        depth_maps = []
        valid_masks = []
        for i, ply in enumerate(ply_files):
            pos_np = _load_gaussian_positions(ply)
            pos = torch.from_numpy(pos_np).to(device=device, dtype=torch.float32)

            depth_i, valid_i = _project_gaussians_to_depth(
                pos, ext[i].to(device), K_pixel[i].to(device),
                out_h=H_out, out_w=W_out,
            )
            depth_maps.append(depth_i)
            valid_masks.append(valid_i)

            n_g = pos.shape[0]
            n_v = int(valid_i.sum())
            d_min = float(depth_i[valid_i].min()) if n_v > 0 else 0.0
            d_med = float(torch.median(depth_i[valid_i])) if n_v > 0 else 0.0
            d_max = float(depth_i[valid_i].max()) if n_v > 0 else 0.0
            log.info(
                f"  face {i}: {n_g} gaussians -> {n_v}/{H_out*W_out} pixels covered "
                f"({100*n_v/(H_out*W_out):.1f}%), depth min={d_min:.3f} "
                f"median={d_med:.3f} max={d_max:.3f}"
            )

        depth_stack = torch.stack(depth_maps, dim=0)  # [N, H, W]
        valid_stack = torch.stack(valid_masks, dim=0)  # [N, H, W]

        # IMAGE convention: [N, H, W, 3], broadcast depth across channels.
        depth_img = depth_stack.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
        valid_img = valid_stack.float().contiguous()

        return io.NodeOutput(depth_img, valid_img, extrinsics, intrinsics)


NODE_CLASS_MAPPINGS = {
    "SharpRefinedDepthFromGaussians": SharpRefinedDepthFromGaussians,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpRefinedDepthFromGaussians": "SHARP Refined Depth from Gaussians",
}
