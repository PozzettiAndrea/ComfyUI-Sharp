"""SharpPanoramaCubeSplit node for ComfyUI-Sharp.

Equirect panorama -> N perspective crops at a fixed pitch x yaw grid.

Parallel to WorldStereo's WorldStereoPanoramaCubeSplit:
  3 pitch levels x N yaw steps at configurable rectangular FOV / aspect.
  Default: 3 x 9 = 27 views at 832x480, 120 deg horizontal x 90 deg
  vertical FOV.

Different from Sharp's existing SamplePanorama (sample_panorama.py):
  - SamplePanorama: square output, FOV-derived dense grid, blends to
    cover the full sphere — flexible for arbitrary FOV/overlap.
  - SharpPanoramaCubeSplit: fixed 3-pitch grid, rectangular aspect,
    matches WorldStereo's anchor-view geometry — for feeding a
    rectilinear depth model (MoGe2 etc.) before merging back to equirect
    via SharpDepthMerge.

Output convention matches the rest of ComfyUI-Sharp:
  face_images (IMAGE [N, H, W, 3]) + extrinsics (EXTRINSICS [N, 4, 4]) +
  intrinsics (INTRINSICS [N, 3, 3]).
"""

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from comfy_api.latest import io

log = logging.getLogger("sharp")


def _rot_around_z(deg: float) -> torch.Tensor:
    """3x3 rotation matrix around the Z axis. World up = +Z convention.

    Matches WorldStereo's `rotate_around_z_axis` in panorama_utils.py so
    the pitch-up / pitch-down hemispheres land at the same world points.
    """
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return torch.tensor([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=torch.float32)


def _look_at_w2c(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """world-to-camera 4x4 matrix from a look-at triple (numpy-free).

    Convention: camera looks down +Z in camera space (matches utils3d
    extrinsics_look_at). Returns a w2c matrix.
    """
    # forward (in world): from eye to target, normalized
    fwd = (target - eye)
    fwd = fwd / (fwd.norm() + 1e-12)
    # right = forward x up
    right = torch.linalg.cross(fwd, up)
    right = right / (right.norm() + 1e-12)
    # recomputed up = right x forward
    new_up = torch.linalg.cross(right, fwd)
    # Camera-to-world rotation: columns are the camera basis in world coords.
    # We want world-to-camera, which is the transpose for pure rotation.
    # In camera space: +X right, +Y down (image convention), +Z forward.
    # So R_c2w has columns = [right, -new_up, fwd]; w2c = transpose.
    R_c2w = torch.stack([right, -new_up, fwd], dim=1)  # [3, 3]
    R_w2c = R_c2w.T

    ext = torch.eye(4, dtype=torch.float32)
    ext[:3, :3] = R_w2c
    ext[:3, 3] = -R_w2c @ eye
    return ext


def _intrinsics_from_fov(fov_x_rad: float, fov_y_rad: float, w: int, h: int) -> torch.Tensor:
    """Pixel-space 3x3 intrinsics from horizontal + vertical FOV."""
    fx = (w / 2.0) / math.tan(fov_x_rad / 2.0)
    fy = (h / 2.0) / math.tan(fov_y_rad / 2.0)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    K = torch.tensor([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1.0],
    ], dtype=torch.float32)
    return K


def _sample_perspective_from_equirect(
    panorama: torch.Tensor,
    ext_w2c: torch.Tensor,
    K_pixel: torch.Tensor,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    """Backward warp: equirect panorama -> a single perspective view.

    Mirrors sample_panorama.py's geometry but with rectangular output
    and an explicit extrinsics/intrinsics input (so the same look-at
    targets used to build ext_w2c here can be passed downstream
    unchanged).
    """
    H_pano, W_pano, _ = panorama.shape
    device = panorama.device

    # Pixel grid in the output perspective image.
    u = torch.arange(out_w, dtype=torch.float32, device=device)
    v = torch.arange(out_h, dtype=torch.float32, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')

    fx, fy = K_pixel[0, 0].item(), K_pixel[1, 1].item()
    cx, cy = K_pixel[0, 2].item(), K_pixel[1, 2].item()

    # Camera-space rays for each output pixel.
    dx = (uu - cx) / fx
    dy = (vv - cy) / fy
    dz = torch.ones_like(dx)
    rays_cam = torch.stack([dx, dy, dz], dim=-1)  # [H, W, 3]
    rays_cam = F.normalize(rays_cam, dim=-1)

    # World rays via camera-to-world rotation.
    R_w2c = ext_w2c[:3, :3].to(device)
    R_c2w = R_w2c.T
    rays_world = torch.einsum('ij,hwj->hwi', R_c2w, rays_cam)  # [H, W, 3]

    # World rays -> equirect sample coords.
    # WorldStereo / HY-World convention: world up = +Z, panorama columns
    # = yaw around Z. Match that here so this node's extrinsics line up
    # with WorldStereo's pano_bank geometry.
    rx, ry, rz = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]
    yaw = torch.atan2(ry, rx)                              # [-pi, pi]
    pitch = torch.asin(torch.clamp(rz, -1.0, 1.0))         # [-pi/2, pi/2]

    eq_x = (yaw / math.pi + 1.0) * (W_pano - 1) / 2.0       # 0..W-1
    eq_y = (0.5 - pitch / math.pi) * (H_pano - 1)           # 0..H-1
    grid_x = eq_x / (W_pano - 1) * 2.0 - 1.0
    grid_y = eq_y / (H_pano - 1) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    pano_nchw = panorama.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    sampled = F.grid_sample(
        pano_nchw, grid,
        mode='bilinear', padding_mode='border', align_corners=True,
    )
    return sampled[0].permute(1, 2, 0)  # [H, W, C]


class SharpPanoramaCubeSplit(io.ComfyNode):
    """Equirect panorama -> N perspective crops at a fixed pitch x yaw grid."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharpPanoramaCubeSplit",
            display_name="Sharp Panorama Cube Split",
            category="SHARP",
            description=(
                "Sample N perspective crops from a 2:1 equirectangular panorama "
                "at a fixed pitch x yaw grid (3 pitch x 9 yaw = 27 views by "
                "default). Matches WorldStereoPanoramaCubeSplit's geometry — "
                "useful as the input stage for a per-face depth model whose "
                "outputs will be stitched back via SharpDepthMerge.\n\n"
                "Different from Sharp's SamplePanorama (which derives the grid "
                "from FOV+overlap, square output): this node has a fixed pitch "
                "grid with rectangular aspect (default 832x480), and matches "
                "the camera/intrinsics convention WorldStereo's pano_bank uses."
            ),
            inputs=[
                io.Image.Input(
                    "panorama",
                    tooltip="Equirectangular RGB panorama, 2:1 aspect. "
                            "[B, H, W, 3] or [H, W, 3] in [0, 1]. First image "
                            "of batch is used."),
                io.Int.Input(
                    "image_w", default=832, min=224, max=2048, step=16,
                    tooltip="Crop width. 832 matches WorldStereo's anchor width."),
                io.Int.Input(
                    "image_h", default=480, min=224, max=2048, step=16,
                    tooltip="Crop height. 480 matches WorldStereo's anchor height."),
                io.Float.Input(
                    "fov_x_deg", default=120.0, min=30.0, max=170.0, step=1.0,
                    tooltip="Horizontal FOV. 120 matches WorldStereo upstream."),
                io.Float.Input(
                    "fov_y_deg", default=90.0, min=30.0, max=170.0, step=1.0,
                    tooltip="Vertical FOV. 90 matches WorldStereo upstream."),
                io.Float.Input(
                    "rot_deg", default=40.0, min=10.0, max=120.0, step=5.0,
                    tooltip="Yaw step between adjacent views. 40 -> 9 views "
                            "per pitch -> 27 total at 3 pitches."),
                io.Float.Input(
                    "pitch_up", default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip="Z component of the upper-hemisphere look-at "
                            "targets. 0.5 matches WorldStereo upstream."),
                io.Float.Input(
                    "pitch_down", default=-0.5, min=-1.0, max=0.0, step=0.05,
                    tooltip="Z component of the lower-hemisphere look-at "
                            "targets. -0.5 matches WorldStereo upstream."),
            ],
            outputs=[
                io.Image.Output(
                    display_name="face_images",
                    tooltip="N perspective crops [N, H, W, 3] in [0, 1]. "
                            "Wire to a depth model (SharpPredictDepth / MoGe2 "
                            "/ etc.)."),
                io.Custom("EXTRINSICS").Output(
                    display_name="extrinsics",
                    tooltip="[N, 4, 4] world-to-camera matrices per face. "
                            "Camera at world origin, look-at along the pitch/"
                            "yaw direction. Pass to SharpDepthMerge."),
                io.Custom("INTRINSICS").Output(
                    display_name="intrinsics",
                    tooltip="[N, 3, 3] pixel-space intrinsics per face. "
                            "Identical across faces (same FOV + size). Pass "
                            "to SharpDepthMerge."),
                io.Float.Output(
                    display_name="fov_x_deg",
                    tooltip="Pass-through. Wire into MoGe2 / depth model's "
                            "fov_x widget so it doesn't have to estimate."),
                io.Int.Output(
                    display_name="num_faces",
                    tooltip="N = 3 * (360 / rot_deg). 27 at defaults."),
            ],
        )

    @classmethod
    def execute(
        cls, panorama: torch.Tensor,
        image_w: int = 832, image_h: int = 480,
        fov_x_deg: float = 120.0, fov_y_deg: float = 90.0,
        rot_deg: float = 40.0,
        pitch_up: float = 0.5, pitch_down: float = -0.5,
    ):
        # Normalize input shape -> [H, W, 3], float in [0, 1].
        if panorama.dim() == 4:
            panorama = panorama[0]
        if panorama.dtype != torch.float32:
            panorama = panorama.float()
        if panorama.shape[-1] == 4:
            panorama = panorama[..., :3]
        # Sharp convention is float [0,1]; guard against accidental uint8 input.
        if panorama.max() > 2.0:
            panorama = panorama / 255.0

        device = panorama.device
        H_pano, W_pano, _ = panorama.shape

        # Build the (3 pitch x N yaw) direction grid.
        # WorldStereo convention: world up = +Z, base directions look at +X
        # rotated around Z. (Eye is world origin; targets are unit-ish
        # points on a sphere.)
        # NOTE: WorldStereo uses [-1, 0, 0] as the base "forward" with +Z up;
        # we mirror that exactly so panorama columns line up the same way.
        base_targets = [
            torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32),
            torch.tensor([-1.0, 0.0, float(pitch_up)], dtype=torch.float32),
            torch.tensor([-1.0, 0.0, float(pitch_down)], dtype=torch.float32),
        ]
        N_yaw = max(1, int(round(360.0 / max(1.0, float(rot_deg)))))
        targets = []
        for base in base_targets:
            targets.append(base)
            for i in range(1, N_yaw):
                R = _rot_around_z(rot_deg * i)
                targets.append((R @ base).contiguous())
        targets = torch.stack(targets, dim=0)  # [N, 3]
        N = targets.shape[0]

        # Same intrinsics for every face (same FOV + same output size).
        fov_x_rad = math.radians(fov_x_deg)
        fov_y_rad = math.radians(fov_y_deg)
        K = _intrinsics_from_fov(fov_x_rad, fov_y_rad, int(image_w), int(image_h))

        # Sample each face.
        eye = torch.zeros(3, dtype=torch.float32)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        face_images = []
        extrinsics = []
        for i in range(N):
            ext_w2c = _look_at_w2c(eye, targets[i], up)
            img = _sample_perspective_from_equirect(
                panorama, ext_w2c, K, int(image_h), int(image_w),
            )
            face_images.append(img)
            extrinsics.append(ext_w2c)

        face_images_t = torch.stack(face_images, dim=0).contiguous()  # [N, H, W, 3]
        ext_t = torch.stack(extrinsics, dim=0).contiguous()  # [N, 4, 4]
        intr_t = K.unsqueeze(0).expand(N, -1, -1).contiguous()  # [N, 3, 3]

        log.info(
            f"[SharpPanoramaCubeSplit] {N} faces ({N_yaw}/pitch x 3 pitches) "
            f"@ {image_w}x{image_h}, fov=({fov_x_deg:.0f},{fov_y_deg:.0f})"
        )

        return io.NodeOutput(face_images_t, ext_t, intr_t, float(fov_x_deg), int(N))


NODE_CLASS_MAPPINGS = {
    "SharpPanoramaCubeSplit": SharpPanoramaCubeSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpPanoramaCubeSplit": "Sharp Panorama Cube Split",
}
