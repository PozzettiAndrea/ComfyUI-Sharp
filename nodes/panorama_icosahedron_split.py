"""SharpPanoramaIcosahedronSplit node for ComfyUI-Sharp.

Equirect panorama -> N square perspective crops at icosahedron vertices,
90° FOV each. Schema mirrors WorldNavPanoramaSplit 1:1 (modulo
`panorama` being plain IMAGE here instead of WorldNav's
WORLDSTEREO_PANORAMA custom type) so workflows port over by editing
only the node-class name.

Geometry:
  - icosahedron_12: 12 base vertices via golden-ratio coords
  - icosahedron_42: subdivide once (add edge midpoints, normalize to
    sphere) -> 12 + 30 = 42 vertices

Why square / icosahedron for SHARP: SHARP's internal resolution is
1536x1536 square and was trained on cube-face-like inputs. The
icosahedron tiling gives uniform sphere coverage (no pole over/under-
sampling like a pitch/yaw grid) at 90° FOV which matches cube-face
geometry exactly.

Implementation is pure numpy + torch. Reuses three helpers from
`panorama_cube_split.py` (`_look_at_w2c`, `_intrinsics_from_fov`,
`_sample_perspective_from_equirect`) so no math is duplicated.

Pole-singularity fix (mirrors WorldNav's): two vertices of the
subdivided icosahedron land at (0, 0, ±1). With the default world-up
of +Z, `forward × up = 0` for those, producing NaN extrinsics. We
detect `|forward · up| > 0.999` per vertex and swap to up=+Y just for
those.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from comfy_api.latest import io

# Re-use the math helpers from the sibling cube-split node — no copy.
from .panorama_cube_split import (
    _look_at_w2c,
    _intrinsics_from_fov,
    _sample_perspective_from_equirect,
)

log = logging.getLogger("sharp")


def _icosahedron_vertices() -> np.ndarray:
    """12 unit-sphere vertices of a regular icosahedron.

    Standard golden-ratio coords. After normalizing to the unit sphere,
    edge length squared between adjacent vertices = (2 / phi) ≈ 1.236.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            ( 0, +1, +phi), ( 0, +1, -phi), ( 0, -1, +phi), ( 0, -1, -phi),
            (+1, +phi,  0), (+1, -phi,  0), (-1, +phi,  0), (-1, -phi,  0),
            (+phi,  0, +1), (+phi,  0, -1), (-phi,  0, +1), (-phi,  0, -1),
        ],
        dtype=np.float32,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    return verts


def _icosahedron_edges(verts: np.ndarray) -> np.ndarray:
    """30 edge (i, j) pairs of the regular icosahedron.

    Derived by picking the 30 shortest pairwise distances out of C(12,2)=66
    — equivalently, the pairs with squared distance ≈ 2/phi after the
    unit-sphere normalization. No hardcoded indices, so this stays correct
    regardless of the vertex ordering in `_icosahedron_vertices`.
    """
    N = len(verts)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            d2 = float(((verts[i] - verts[j]) ** 2).sum())
            pairs.append((d2, i, j))
    pairs.sort()
    # The icosahedron has 30 edges; threshold by taking exactly the 30
    # closest pairs. Sanity-check: the gap between edge-distance and
    # non-edge-distance should be large (>1.5x).
    edge_d2 = pairs[29][0]
    non_edge_d2 = pairs[30][0]
    assert non_edge_d2 > edge_d2 * 1.5, (
        f"icosahedron edge classification ambiguous: "
        f"edge_d2={edge_d2:.4f}, non_edge_d2={non_edge_d2:.4f}"
    )
    return np.array([(i, j) for _, i, j in pairs[:30]], dtype=np.int32)


def _subdivide_icosahedron(verts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Add edge midpoints (normalized to the unit sphere) → 42 vertices."""
    midpoints = (verts[edges[:, 0]] + verts[edges[:, 1]]) / 2.0
    midpoints /= np.linalg.norm(midpoints, axis=1, keepdims=True)
    return np.concatenate([verts, midpoints.astype(np.float32)], axis=0)


def _icosahedron_split_vertices(subdivision: str) -> np.ndarray:
    """Return [N, 3] unit-sphere target points for the chosen subdivision.

    Sorted top-to-bottom in a spiral:
      - Primary key: -pitch  (= -asin(z), so highest latitude first)
      - Secondary key: yaw   (= atan2(y, x), going around -π → π at each ring)
    So index 0 is the topmost vertex (north pole if present), the next
    several indices walk around the next ring down, then the ring after
    that, ..., and the last index is the bottommost vertex.
    """
    base = _icosahedron_vertices()
    if subdivision == "icosahedron_12":
        verts = base
    elif subdivision == "icosahedron_42":
        edges = _icosahedron_edges(base)
        verts = _subdivide_icosahedron(base, edges)
    else:
        raise ValueError(f"unknown subdivision={subdivision!r}")

    pitch = np.arcsin(np.clip(verts[:, 2], -1.0, 1.0))     # [-π/2, π/2]
    yaw = np.arctan2(verts[:, 1], verts[:, 0])             # [-π, π]
    # lexsort uses the LAST key as primary. Round pitch to 4 decimals so
    # within-ring ties (same latitude, different yaw) get grouped before
    # the secondary yaw sort kicks in — avoids floating-point jitter from
    # putting nominally-same-pitch vertices in different rings.
    pitch_key = -np.round(pitch, decimals=4)
    order = np.lexsort((yaw, pitch_key))
    return verts[order].astype(np.float32)


def _build_extrinsics_with_pole_fix(targets: torch.Tensor) -> torch.Tensor:
    """Build per-vertex world-to-camera matrices, swapping up at poles.

    `targets`: [N, 3] unit-sphere vectors (look-at targets from world
    origin). World up default = +Z. For vertices within ~2.5° of ±Z,
    swap up to +Y to avoid the cross-product singularity.

    Always computes on CPU regardless of input device — 42 small
    look-at matrices, the device round-trip dominates any GPU win.
    Caller .to(device) on the return.
    """
    targets = targets.detach().cpu().float()
    N = targets.shape[0]
    eye = torch.zeros(3, dtype=torch.float32)
    up_default = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    up_fallback = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    cos_thresh = 0.999

    out = torch.empty(N, 4, 4, dtype=torch.float32)
    cos_with_up = torch.abs(targets @ up_default)
    for i in range(N):
        up = up_fallback if float(cos_with_up[i]) > cos_thresh else up_default
        out[i] = _look_at_w2c(eye, targets[i], up)
    return out


def _hsv_color_bgr(idx: int, total: int) -> tuple[int, int, int]:
    """Distinct BGR color per face index via HSV color wheel.

    Mirrors WorldNavPanoramaSplit's `_hsv_color`. cv2 expects BGR.
    Hue in OpenCV is uint8 in [0, 179]; cycle through with full
    saturation + value for maximum contrast.
    """
    import cv2
    hue = int(round(180.0 * idx / max(total, 1))) % 180
    hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _make_debug_overlay(
    panorama: np.ndarray,
    extrinsics: torch.Tensor,
) -> np.ndarray:
    """Draw each face's frustum edges on the panorama.

    For each face: sample N=64 points along each of the 4 edges of the
    90° camera image plane in camera space (corners at z=1: (±1, ±1, 1)),
    transform to world via R_c2w = R_w2c^T, normalize to the unit
    sphere, convert to (yaw, pitch), then to ERP pixel coords. Draw a
    polyline per edge with a distinct HSV-wheel hue. Handle azimuth
    wraparound by splitting any edge that crosses the ±π seam into
    separate segments.

    Sharp's coordinate convention (matches `_sample_perspective_from_equirect`):
        world up = +Z
        yaw    = atan2(ry, rx)         around Z
        pitch  = asin(rz)              from XY plane
        u_erp  = ((yaw / π) * 0.5 + 0.5) * W
        v_erp  = (0.5 - pitch / π) * H

    Args:
        panorama: (H, W, 3) uint8 RGB.
        extrinsics: (N, 4, 4) world-to-camera, CPU tensor.

    Returns:
        (H, W, 3) uint8 RGB with N face boundaries drawn on it.
    """
    import cv2
    H, W = panorama.shape[:2]
    debug_img = panorama.copy()
    N = int(extrinsics.shape[0])
    ext_np = extrinsics.detach().cpu().numpy().astype(np.float32)

    S = 64
    edge_corners = [
        ([-1.0, -1.0], [+1.0, -1.0]),  # top
        ([+1.0, -1.0], [+1.0, +1.0]),  # right
        ([+1.0, +1.0], [-1.0, +1.0]),  # bottom
        ([-1.0, +1.0], [-1.0, -1.0]),  # left
    ]

    for i in range(N):
        R_w2c = ext_np[i, :3, :3]
        R_c2w = R_w2c.T  # camera -> world (orthonormal rotation)

        color = _hsv_color_bgr(i, N)
        for (p0, p1) in edge_corners:
            t = np.linspace(0.0, 1.0, S, dtype=np.float32)
            xs = p0[0] + t * (p1[0] - p0[0])
            ys = p0[1] + t * (p1[1] - p0[1])
            cam_dirs = np.stack(
                [xs, ys, np.ones_like(xs)], axis=-1,
            )  # (S, 3)
            # row * R_c2w^T = world-direction row (same as cam_col → R_c2w @ cam_col)
            world_dirs = cam_dirs @ R_c2w.T
            world_dirs /= np.maximum(
                np.linalg.norm(world_dirs, axis=-1, keepdims=True), 1e-12,
            )
            # Sharp convention: world up = +Z, yaw around Z, pitch from XY.
            yaw = np.arctan2(world_dirs[:, 1], world_dirs[:, 0])         # [-π, π]
            pitch = np.arcsin(np.clip(world_dirs[:, 2], -1.0, 1.0))       # [-π/2, π/2]
            u = ((yaw / np.pi) * 0.5 + 0.5) * W
            v = (0.5 - pitch / np.pi) * H
            pts = np.stack([u, v], axis=-1).astype(np.float32)

            # Split at azimuth wraparound: any consecutive pair with
            # |Δu| > W/2 wrapped the ±π seam.
            du = np.abs(np.diff(pts[:, 0]))
            breaks = np.where(du > W * 0.5)[0]
            segments = np.split(pts, breaks + 1) if len(breaks) else [pts]
            for seg in segments:
                if len(seg) < 2:
                    continue
                seg_int = seg.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    debug_img, [seg_int], isClosed=False,
                    color=color, thickness=2, lineType=cv2.LINE_AA,
                )
    return debug_img


class SharpPanoramaIcosahedronSplit(io.ComfyNode):
    """Equirect panorama -> N square perspective crops at icosahedron vertices."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharpPanoramaIcosahedronSplit",
            display_name="Sharp Panorama Icosahedron Split",
            category="SHARP",
            description=(
                "Split an equirectangular panorama into N rectilinear views "
                "(90° fov each) at icosahedron vertices. Pipe the views "
                "through any rectilinear depth model (e.g. SHARP / MoGe2), "
                "then merge the per-view depths back into an equirect depth "
                "map via SharpDepthMerge.\n\n"
                "1:1 schema mirror of WorldNavPanoramaSplit (modulo "
                "panorama input being plain IMAGE here, not WorldStereo's "
                "WORLDSTEREO_PANORAMA custom type)."
            ),
            inputs=[
                io.Image.Input(
                    "panorama",
                    tooltip="Equirectangular panorama (2:1). [B, H, W, 3] "
                            "or [H, W, 3]. Values passed through as-is — "
                            "feed RGB in [0, 1] OR a depth panorama in "
                            "meters (e.g. SharpDepthMerge.depth, used to "
                            "re-cut the merged depth into per-face crops). "
                            "The bilinear sampler is linear so either "
                            "value range works; the debug overlay "
                            "auto-normalizes by p99 for visualization."),
                io.Int.Input(
                    "resolution", default=512, min=128, max=2048, step=64,
                    tooltip="Per-face image resolution (square). Default 512 "
                            "matches WorldNavPanoramaSplit."),
                io.Combo.Input(
                    "subdivision",
                    options=["icosahedron_12", "icosahedron_42"],
                    default="icosahedron_42",
                    tooltip="Tiling density. 12 = base icosahedron (faster), "
                            "42 = subdivided (better polar coverage)."),
                io.Float.Input(
                    "fov_deg", default=90.0, min=30.0, max=170.0, step=1.0,
                    tooltip="Per-face FOV in degrees (square, used for both "
                            "horizontal and vertical). Default 90° — matches "
                            "WorldNavPanoramaSplit, SHARP's training "
                            "distribution sweet-spot, and gives ~2× sphere "
                            "over-coverage with icosahedron_12 (≈7× with _42).\n\n"
                            "Coverage trade-off: narrower FOV may leave gaps "
                            "in the sphere (icosahedron_12 needs ≥85° to "
                            "cover the full sphere without gaps; icosahedron_42 "
                            "is safe down to ~55°). Wider FOV gives more "
                            "redundancy but per-face quality drops past ~110° "
                            "as SHARP goes out-of-distribution and rectilinear "
                            "distortion gets severe near face edges."),
                io.Boolean.Input(
                    "use_gpu", default=True, optional=True,
                    tooltip="Batched panorama->face resampling via "
                            "torch.nn.functional.grid_sample (bilinear) on "
                            "CUDA. Falls back to CPU if CUDA isn't "
                            "available. Math identical within fp32 round-off."),
                io.Boolean.Input(
                    "convert_distance_to_planar", default=False, optional=True,
                    tooltip=(
                        "ONLY enable when the input panorama is a DEPTH map "
                        "in equirect ray-distance convention (e.g. the output "
                        "of SharpDepthMerge, which stores Euclidean distance "
                        "from the camera center along each equirect pixel's "
                        "ray direction).\n\n"
                        "Equirect depth is naturally a 'ray-distance field' "
                        "(per direction), but downstream consumers like "
                        "SharpPredictGaussiansFromMetricDepth expect per-face "
                        "PLANAR Z (depth along the face's optical axis). At "
                        "the corner of a 90° face the two differ by ~73% — "
                        "feeding ray-distance where planar Z is expected "
                        "causes the same world point to land at different "
                        "3D positions when viewed from different faces.\n\n"
                        "When True: after bilinear sampling each face, "
                        "multiply by the per-pixel cos-map "
                        "  cos_map[u, v] = 1 / sqrt(((u-cx)/fx)² + "
                        "((v-cy)/fy)² + 1)\n"
                        "which depends only on K (shared across all faces). "
                        "Output is per-face planar Z, ready to feed SHARP.\n\n"
                        "Leave False (default) for RGB panoramas — color is "
                        "per-direction and doesn't need conversion."
                    )),
            ],
            outputs=[
                io.Image.Output(display_name="face_images"),
                io.Custom("EXTRINSICS").Output(display_name="extrinsics"),
                io.Custom("INTRINSICS").Output(display_name="intrinsics"),
                io.Float.Output(display_name="fov_x_deg"),
                io.Image.Output(display_name="debug_image"),
            ],
        )

    @classmethod
    def execute(
        cls, panorama: torch.Tensor,
        resolution: int = 512,
        subdivision: str = "icosahedron_42",
        fov_deg: float = 90.0,
        use_gpu: bool = True,
        convert_distance_to_planar: bool = False,
    ):
        # Normalize input shape -> [H, W, 3], float. Values are passed
        # through as-is — no uint8-detection rescale. Caller is responsible
        # for whatever value range is meaningful (RGB in [0, 1] for image
        # inputs, meters for depth inputs, etc.). The bilinear sampler
        # is linear in input values either way.
        pano = panorama
        if pano.dim() == 4:
            pano = pano[0]
        if pano.dtype != torch.float32:
            pano = pano.float()
        if pano.shape[-1] == 4:
            pano = pano[..., :3]

        # Horizontal flip — reconcile input pano convention (yaw decreases
        # left-to-right = standard photo / generative outputs) with the
        # sampler's internal convention (yaw increases left-to-right =
        # WorldStereo). Without this, face images come out mirrored
        # relative to the input.
        pano = torch.flip(pano, dims=[-2])  # W axis on [H, W, C]

        H_pano, W_pano, _ = pano.shape
        # Hard 2:1 aspect check — same as WorldNavPanoramaSplit.
        aspect = W_pano / H_pano
        if abs(aspect - 2.0) > 0.01:
            raise ValueError(
                f"SharpPanoramaIcosahedronSplit: panorama must be 2:1 "
                f"equirect, got {W_pano}x{H_pano} (ratio {aspect:.3f}). "
                f"Crop / pad to 2:1 first."
            )

        device = pano.device
        if use_gpu and torch.cuda.is_available() and device.type != "cuda":
            pano = pano.cuda()
            device = pano.device

        targets_np = _icosahedron_split_vertices(subdivision)
        targets = torch.from_numpy(targets_np).to(device=device, dtype=torch.float32)
        N = targets.shape[0]

        # Per-vertex extrinsics with pole-singularity fix.
        extrinsics = _build_extrinsics_with_pole_fix(targets).to(device)

        # Same intrinsics for every face (90° FOV, square).
        fov_x_deg = float(fov_deg)
        fov_y_deg = float(fov_deg)
        K = _intrinsics_from_fov(
            math.radians(fov_x_deg), math.radians(fov_y_deg),
            int(resolution), int(resolution),
        ).to(device)

        face_images = []
        for i in range(N):
            img = _sample_perspective_from_equirect(
                pano, extrinsics[i], K, int(resolution), int(resolution),
            )
            face_images.append(img)

        face_images_t = torch.stack(face_images, dim=0).contiguous()  # [N, R, R, 3]
        intrinsics_t = K.unsqueeze(0).expand(N, -1, -1).contiguous()  # [N, 3, 3]
        extrinsics_t = extrinsics.contiguous()  # [N, 4, 4]

        # Optional ray-distance → per-face planar-Z conversion. Equirect depth
        # is naturally per-direction ray-distance (Euclidean from camera
        # center); SHARP's NDC unprojection expects planar Z (along the face's
        # optical axis). Convert by multiplying with cos(angle from optical
        # axis) = 1 / sqrt(((u-cx)/fx)² + ((v-cy)/fy)² + 1). Same map for
        # every face (depends only on K).
        if convert_distance_to_planar:
            R = int(resolution)
            uu = torch.arange(R, dtype=torch.float32, device=device)
            vv = torch.arange(R, dtype=torch.float32, device=device)
            uu_g, vv_g = torch.meshgrid(uu, vv, indexing="xy")  # (R, R)
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            x_cam = (uu_g - cx) / fx
            y_cam = (vv_g - cy) / fy
            ray_norm = torch.sqrt(x_cam * x_cam + y_cam * y_cam + 1.0)
            cos_map = 1.0 / ray_norm  # (R, R), values in (~0.577, 1] for 90° fov
            face_images_t = face_images_t * cos_map.unsqueeze(0).unsqueeze(-1)
            log.info(
                f"[SharpPanoramaIcosahedronSplit] applied ray-distance → "
                f"planar-Z conversion: cos_map min={float(cos_map.min()):.4f} "
                f"max={float(cos_map.max()):.4f} (factor at corners depends "
                f"on FOV)"
            )

        # Debug overlay: panorama with each face's frustum edges drawn on it.
        # uint8 round-trip needed for cv2.polylines; pano values may be RGB
        # in [0, 1] OR depth in meters OR anything else linear, so normalize
        # by p99 (clip outliers, divide by p99) before the uint8 cast. For
        # standard [0, 1] RGB this is a near-identity (p99 ≈ 1.0). For depth
        # in meters it gives a sensible grayscale visualization without
        # crushing dynamic range.
        # `pano` here is the FLIPPED version (post-sampling-prep flip).
        # `_make_debug_overlay` computes frustum edge u-coords in Sharp's
        # internal convention, which matches the flipped pano. Drawing
        # the overlay on the flipped pano gives correct edge positions,
        # but the resulting debug image looks horizontally mirrored
        # relative to the user's original input. To present a debug
        # image that matches the user's INPUT panorama (with the face
        # frustum edges drawn at the visually correct positions on it),
        # we flip the overlay output back along the W axis.
        _pano_np = pano.detach().cpu().numpy()
        _p99 = float(np.nanquantile(_pano_np, 0.99))
        _p99 = max(_p99, 1e-6)
        pano_uint8 = (
            np.clip(_pano_np / _p99, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        debug_np = _make_debug_overlay(pano_uint8, extrinsics)  # (H, W, 3) uint8
        # Flip the overlay back to the user's input pano convention.
        debug_np = np.ascontiguousarray(debug_np[:, ::-1])
        debug_image = (
            torch.from_numpy(debug_np.astype(np.float32) / 255.0).unsqueeze(0)
        )  # [1, H, W, 3]

        log.info(
            f"[SharpPanoramaIcosahedronSplit] {N} faces "
            f"({subdivision}) @ {resolution}x{resolution}, fov={fov_x_deg:.1f}°"
        )

        return io.NodeOutput(
            face_images_t, extrinsics_t, intrinsics_t,
            float(fov_x_deg), debug_image,
        )


NODE_CLASS_MAPPINGS = {
    "SharpPanoramaIcosahedronSplit": SharpPanoramaIcosahedronSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpPanoramaIcosahedronSplit": "Sharp Panorama Icosahedron Split",
}
