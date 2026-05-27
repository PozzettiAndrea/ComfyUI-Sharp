"""SharpBuildMesh — equirect depth + panorama → trimesh.Trimesh.

Pure-geometry mirror of WorldNavBuildMesh. Consumes an equirect panorama and
an equirect distance map (e.g. from SharpDepthMerge) and produces a triangle
mesh by deprojecting each panorama pixel along its spherical ray.

Inputs:
  - panorama IMAGE        — equirect RGB (2:1)
  - depth IMAGE           — equirect distance map (only first channel used)
  - sky_mask MASK         — exclude sky pixels (optional)
  - valid_mask MASK       — exclude LSMR-invalid pixels (optional)
  - scene_cap_depth FLOAT — hard clip on far depth (default 8m, 0 disables)

Outputs:
  - mesh (TRIMESH)            — trimesh.Trimesh with vertex colors from the panorama
  - mesh_preview (IMAGE)      — 3 axis-plane cross-sections (pyvista wireframe)
  - global_median_depth FLOAT — median of valid depth pixels post-clipping

Heavy deps (open3d, pyvista, trimesh, utils3d) are lazy-imported. The
vendored MoGe `convert_rgbd2mesh_panorama` does the actual mesh build —
same code path as WorldNavBuildMesh and HY-World 2.0 upstream.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io


def _p(msg: str) -> None:
    print(f"[SharpBuildMesh] {msg}", file=sys.stderr, flush=True)


def _make_pbar(total: int = 100):
    try:
        import comfy.utils
        return comfy.utils.ProgressBar(total)
    except Exception:
        class _Noop:
            def update_absolute(self, *_a, **_k): pass
        return _Noop()


def _normalize_panorama(panorama_pil):
    from PIL import Image
    full_img = panorama_pil.convert("RGB")
    if full_img.size[1] > 1920:
        _p(f"panorama {full_img.size} > 1920 tall → resizing to 3840×1920")
        full_img = full_img.resize((3840, 1920), resample=Image.Resampling.BICUBIC)
    return full_img


def _render_mesh_preview(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    width: int = 1536,
    height: int = 512,
) -> np.ndarray:
    """Three orthographic wireframe cross-sections (YZ at x=0, XZ at y=0, XY at z=0).

    Each cut keeps the half opposite the camera so we see the cut face. Falls
    back to a placeholder image if pyvista isn't importable.
    """
    from PIL import Image, ImageDraw
    try:
        import os as _os
        _os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
        import pyvista as pv
    except Exception as e:
        _p(f"pyvista import failed: {e!r} — placeholder preview")
        ph = Image.new("RGB", (width, height), (24, 24, 28))
        d = ImageDraw.Draw(ph)
        d.text((20, 20), f"pyvista not installed: {e}\nRun `cds install` to add it.",
               fill=(220, 220, 220))
        return np.array(ph, dtype=np.uint8)

    F_count = int(triangles.shape[0])
    faces_flat = np.concatenate(
        [np.full((F_count, 1), 3, dtype=np.int64), triangles.astype(np.int64)],
        axis=1,
    ).flatten()
    poly = pv.PolyData(vertices.astype(np.float32), faces_flat)

    bounds = poly.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1e-6)
    R = span * 1.8

    cuts = [
        ("YZ slice (x = 0)", ( 1.0, 0.0, 0.0), ( R, 0, 0), (0, 0, 1)),
        ("XZ slice (y = 0)", ( 0.0, 1.0, 0.0), (0,  R, 0), (0, 0, 1)),
        ("XY slice (z = 0)", ( 0.0, 0.0, 1.0), (0, 0,  R), (0, 1, 0)),
    ]

    plotter = pv.Plotter(
        off_screen=True,
        shape=(1, 3),
        window_size=(int(width), int(height)),
        border=True,
        border_color="gray",
    )
    plotter.set_background((0.09, 0.09, 0.11))

    for i, (title, normal, cam_pos, view_up) in enumerate(cuts):
        plotter.subplot(0, i)
        clipped = poly.clip(normal=normal, origin=(0.0, 0.0, 0.0), invert=True)
        if clipped.n_cells > 0:
            plotter.add_mesh(
                clipped,
                color=(0.55, 0.72, 0.95),
                show_edges=True,
                edge_color=(0.18, 0.28, 0.45),
                line_width=0.3,
                lighting=True,
                ambient=0.25,
                diffuse=0.7,
            )
        plotter.add_mesh(pv.Sphere(radius=max(span * 0.01, 1e-3)),
                         color=(1.0, 0.85, 0.3))
        plotter.camera_position = [tuple(cam_pos), (0.0, 0.0, 0.0), tuple(view_up)]
        plotter.add_text(title, font_size=10, color="white", position="upper_left")

    img = plotter.screenshot(return_img=True)
    plotter.close()
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.shape[:2] != (int(height), int(width)):
        img = np.array(
            Image.fromarray(img).resize((int(width), int(height)), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
    return img


class SharpBuildMesh(io.ComfyNode):
    """Equirect panorama + depth → trimesh.Trimesh (+ 3-viewport preview)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharpBuildMesh",
            display_name="SHARP Build Mesh",
            category="SHARP",
            description=(
                "Equirect panorama + equirect depth → trimesh.Trimesh. "
                "Pure geometry: no AI models loaded. Mirrors WorldNavBuildMesh "
                "and HY-World 2.0's traj_generate.py mesh-build stage; calls the "
                "vendored MoGe `convert_rgbd2mesh_panorama` under the hood. "
                "Bonus mesh_preview output: 3 orthographic wireframe cross-"
                "sections so you can sanity-check geometry before downstream ops."
            ),
            inputs=[
                io.Image.Input(
                    "panorama",
                    tooltip="Equirectangular RGB panorama (2:1 aspect). "
                            "Native IMAGE input — no panorama wrapping needed."),
                io.Image.Input(
                    "depth",
                    tooltip="Equirect distance map from SharpDepthMerge "
                            "(or any 2:1 depth tensor). Only the first "
                            "channel is used."),
                io.Mask.Input(
                    "sky_mask",
                    optional=True,
                    tooltip="Sky pixels to exclude from the mesh. Same H×W as depth."),
                io.Mask.Input(
                    "valid_mask",
                    optional=True,
                    tooltip="Valid-pixel mask (e.g. SharpDepthMerge's valid_mask "
                            "output). Untrusted pixels skipped during mesh build."),
                io.Float.Input(
                    "scene_cap_depth",
                    default=8.0, min=0.0, max=1000.0, step=0.5,
                    optional=True,
                    tooltip="Hard upper cap on depth in meters. Caps far "
                            "background / sky so the mesh extent stays bounded. "
                            "0 disables. Default 8m matches HY-World's typical "
                            "outdoor contract."),
                io.Int.Input("seed", default=1024, min=0, max=2**31 - 1),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.Image.Output(display_name="mesh_preview"),
                io.Float.Output(display_name="global_median_depth"),
            ],
        )

    @classmethod
    def execute(cls, panorama, depth, sky_mask=None, valid_mask=None,
                scene_cap_depth=8.0, seed=1024):
        from PIL import Image
        from ._vendor.worldgen.src.panorama_utils import (
            convert_rgbd2mesh_panorama,
            spherical_uv_to_directions,
        )
        from utils3d.numpy.maps import uv_map as _uv_map, depth_map_edge as _depth_map_edge
        import trimesh as _tm

        torch.manual_seed(int(seed))
        np.random.seed(int(seed) & 0xFFFFFFFF)

        # --- panorama → PIL ---
        arr = panorama.detach().cpu().numpy() if isinstance(panorama, torch.Tensor) else np.asarray(panorama)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
        full_img = _normalize_panorama(Image.fromarray(arr))

        # --- depth → [H, W] float ---
        d = depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth)
        if d.ndim == 4:
            d = d[0]
        if d.ndim == 3:
            d = d[..., 0]
        depth_np = d.astype(np.float32)
        H, W = depth_np.shape[:2]

        # --- masks → bool [H, W] ---
        sky_mask_np: Optional[np.ndarray] = None
        if sky_mask is not None:
            sm = sky_mask.detach().cpu().numpy() if isinstance(sky_mask, torch.Tensor) else np.asarray(sky_mask)
            if sm.ndim == 3:
                sm = sm[0]
            sky_mask_np = (sm > 0.5).astype(bool)

        valid_mask_np: Optional[np.ndarray] = None
        if valid_mask is not None:
            vm = valid_mask.detach().cpu().numpy() if isinstance(valid_mask, torch.Tensor) else np.asarray(valid_mask)
            if vm.ndim == 3:
                vm = vm[0]
            valid_mask_np = (vm > 0.5).astype(bool)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pbar = _make_pbar(100)

        def _stage(pct, msg):
            _p(f"[{pct:3d}%] {msg}")
            pbar.update_absolute(pct, 100)

        _stage(2, f"depth {depth_np.shape}, sky={'y' if sky_mask_np is not None else 'n'}, "
                  f"valid={'y' if valid_mask_np is not None else 'n'}, "
                  f"scene_cap={float(scene_cap_depth or 0.0):.2f}m")

        # --- equirect rays from depth shape ---
        _stage(10, f"equirect rays from depth {(H, W)}")
        uv = _uv_map(H, W)
        rays_np = spherical_uv_to_directions(uv).astype(np.float32)
        d_min = float(np.nanmin(depth_np))
        d_med = float(np.nanmedian(depth_np))
        d_max = float(np.nanmax(depth_np))
        _p(f"  depth stats: min={d_min:.3f} median={d_med:.3f} max={d_max:.3f}")

        full_distance = torch.as_tensor(depth_np, dtype=torch.float32, device=dev)
        full_rays = torch.as_tensor(rays_np, dtype=torch.float32, device=dev)

        # --- depth post-process: edge mask + sky + quantile-99 clip ---
        _stage(25, "post-process: edge + sky + q99 clip")
        edge_mask = torch.from_numpy(
            _depth_map_edge(full_distance.cpu().numpy(), rtol=0.1)
        ).bool()
        if sky_mask_np is not None:
            sky_t = torch.from_numpy(sky_mask_np)
        else:
            sky_t = torch.zeros((H, W), dtype=torch.bool)
        if valid_mask_np is not None:
            invalid_init = torch.from_numpy(~valid_mask_np)
            _p(f"  external valid_mask: {int(valid_mask_np.sum())} / {valid_mask_np.size}")
        else:
            invalid_init = torch.zeros((H, W), dtype=torch.bool)

        sky_for_depth = sky_t
        if sky_for_depth.shape != edge_mask.shape:
            _p(f"  resizing sky mask {tuple(sky_for_depth.shape)} → {tuple(edge_mask.shape)}")
            sky_for_depth = F.interpolate(
                sky_for_depth[None, None].float(),
                size=edge_mask.shape, mode="nearest",
            )[0, 0].bool()
        full_mask = (sky_for_depth | edge_mask | invalid_init).to(dev)
        valid_pix = full_distance[~full_mask]
        if valid_pix.numel() == 0:
            raise RuntimeError("SharpBuildMesh: all pixels masked out — nothing to mesh")
        max_d = torch.quantile(valid_pix, q=0.99).item()
        _p(f"  q99 depth = {max_d:.3f}, clipping to [0, {max_d:.3f}]")
        full_distance = torch.clip(full_distance, 0, max_d)

        if scene_cap_depth is not None and scene_cap_depth > 0:
            cap = float(scene_cap_depth)
            n_capped = int((full_distance > cap).sum().item())
            _p(f"  scene_cap_depth={cap:.2f}m → clipping {n_capped} pixels")
            full_distance = torch.clip(full_distance, 0, cap)

        # --- resize to mesh resolution (960×1920) and build mesh ---
        _stage(45, "convert_rgbd2mesh_panorama @ 960×1920")
        mesh_h, mesh_w = 960, 1920
        img_resized = full_img.resize((mesh_w, mesh_h), resample=Image.Resampling.BICUBIC)
        distance_resized = F.interpolate(
            full_distance[None, None], size=(mesh_h, mesh_w), mode="nearest",
        )[0, 0]
        rays_resized = F.interpolate(
            full_rays.permute(2, 0, 1)[None], size=(mesh_h, mesh_w), mode="bilinear",
        )[0].permute(1, 2, 0)
        sky_mask_resized = F.interpolate(
            sky_t.float()[None, None].to(dev), size=(mesh_h, mesh_w), mode="nearest",
        )[0, 0].bool()
        _p(f"  resized to {mesh_h}×{mesh_w}; sky pixels @ mesh res: {int(sky_mask_resized.sum())}")

        rgb_t = torch.as_tensor(np.array(img_resized) / 255.0, dtype=torch.float32)
        o3d_mesh = convert_rgbd2mesh_panorama(
            rgb=rgb_t,
            distance=distance_resized.to(dev),
            rays=rays_resized.to(dev),
            excluded_region_mask=sky_mask_resized.to(dev),
            device="cuda" if dev.type == "cuda" else "cpu",
        )

        # --- open3d → trimesh.Trimesh (carry vertex colors) ---
        _stage(85, "open3d → trimesh.Trimesh")
        vertices = np.asarray(o3d_mesh.vertices, dtype=np.float32)
        triangles = np.asarray(o3d_mesh.triangles, dtype=np.int32)
        global_median_depth = float(torch.median(full_distance[~full_mask]).item())
        vertex_colors_rgba = None
        if o3d_mesh.has_vertex_colors():
            vc = np.asarray(o3d_mesh.vertex_colors)
            if vc.shape[0] == vertices.shape[0]:
                vc = np.clip(vc, 0.0, 1.0)
                vertex_colors_rgba = np.empty((vc.shape[0], 4), dtype=np.uint8)
                vertex_colors_rgba[:, :3] = (vc * 255.0 + 0.5).astype(np.uint8)
                vertex_colors_rgba[:, 3] = 255
            else:
                _p(f"  WARN: vertex_colors {vc.shape[0]} != verts {vertices.shape[0]} — dropping")
        tm_mesh = _tm.Trimesh(
            vertices=vertices,
            faces=triangles,
            vertex_colors=vertex_colors_rgba,
            process=False,
        )
        _p(f"  mesh: {len(vertices)} verts, {len(triangles)} faces; "
           f"global_median_depth={global_median_depth:.3f}; "
           f"vertex_colors={'yes' if vertex_colors_rgba is not None else 'no'}")

        # --- preview (3-viewport orthographic wireframe) ---
        _stage(92, "rendering mesh_preview wireframe")
        preview_np = _render_mesh_preview(vertices, triangles, width=1536, height=512)
        preview_t = torch.from_numpy(preview_np.astype(np.float32) / 255.0).unsqueeze(0)

        _stage(100, "done")
        return io.NodeOutput(tm_mesh, preview_t, global_median_depth)


NODE_CLASS_MAPPINGS = {
    "SharpBuildMesh": SharpBuildMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpBuildMesh": "SHARP Build Mesh",
}
