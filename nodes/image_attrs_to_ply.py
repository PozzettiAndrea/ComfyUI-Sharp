"""SharpImageAttrsToPLY — write a PLY from a MULTIBAND_IMAGE of gaussian attrs.

Consumes the `first_layer_gaussian_attrs` / `second_layer_gaussian_attrs`
output of `SharpPredictGaussianAttrs` (or any 14-channel MULTIBAND_IMAGE
in SHARP's gaussian schema) and emits a PLY file in world coordinates.

Channel order is fixed by `SharpPredictGaussianAttrs.GAUSS_ATTR_CHANNEL_NAMES`:

  0:  position_x   (NDC)
  1:  position_y   (NDC)
  2:  position_z   (NDC / camera-space z)
  3:  scale_x      (Sigma-singular value, world units after unprojection)
  4:  scale_y
  5:  scale_z
  6:  quaternion_w
  7:  quaternion_x
  8:  quaternion_y
  9:  quaternion_z
  10: color_r      (linear or SH-DC)
  11: color_g
  12: color_b
  13: opacity      (post-activation, in [0, 1])

We rebuild a `Gaussians3D` namedtuple in NDC, then call SHARP's
`unproject_gaussians` to transform to world coordinates using the
per-face extrinsics + intrinsics. Output convention matches SharpPredict:
single batch -> one .ply; N>1 batch -> folder of NNN.ply.
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from comfy_api.latest import io

log = logging.getLogger("sharp")

try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


def _p(msg: str) -> None:
    print(f"[SharpImageAttrsToPLY] {msg}", file=sys.stderr, flush=True)


# Channel offsets within the 14-channel multiband — must match
# `predict_gaussian_attrs.GAUSS_ATTR_CHANNEL_NAMES`.
POS = slice(0, 3)
SCALE = slice(3, 6)
QUAT = slice(6, 10)
COLOR = slice(10, 13)
OPACITY = 13


class SharpImageAttrsToPLY(io.ComfyNode):
    """MULTIBAND_IMAGE of gaussian attrs + camera -> PLY file."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharpImageAttrsToPLY",
            display_name="SHARP Image + Attrs to PLY",
            category="SHARP",
            description=(
                "Take a 14-channel MULTIBAND_IMAGE of per-pixel gaussian "
                "attributes (e.g. from SharpPredictGaussianAttrs's "
                "first_layer_gaussian_attrs output) + the matching camera "
                "extrinsics/intrinsics and write a 3D Gaussian Splatting "
                "PLY file. The multiband stores positions in NDC; this "
                "node unprojects to world coords via SHARP's "
                "unproject_gaussians before saving."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("MULTIBAND_IMAGE").Input(
                    "gaussian_attrs",
                    tooltip="14-channel multiband (samples [B, 14, H, W]) "
                            "with channel order position_xyz, scale_xyz, "
                            "quat_wxyz, color_rgb, opacity. Typically the "
                            "first_layer_gaussian_attrs or "
                            "second_layer_gaussian_attrs output of "
                            "SharpPredictGaussianAttrs."),
                io.Custom("EXTRINSICS").Input(
                    "extrinsics",
                    tooltip="Per-batch extrinsics matching the multiband "
                            "batch (4×4 each). From "
                            "SharpPredictGaussianAttrs.extrinsics."),
                io.Custom("INTRINSICS").Input(
                    "intrinsics",
                    tooltip="Per-batch intrinsics rescaled to the gaussian "
                            "grid (3×3 or 4×4 pixel-K for H×W). From "
                            "SharpPredictGaussianAttrs.intrinsics."),
                io.String.Input(
                    "output_prefix", default="sharp_attrs",
                    optional=True,
                    tooltip="Basename for the output PLY (single) or batch "
                            "folder (N>1)."),
                io.Float.Input(
                    "focal_length_px", default=0.0, min=0.0, max=8192.0,
                    step=1.0, optional=True,
                    tooltip="Override focal length in pixels saved into the "
                            "PLY metadata. 0 = use intrinsics[0, 0] directly."),
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
        gaussian_attrs: dict,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        output_prefix: str = "sharp_attrs",
        focal_length_px: float = 0.0,
    ):
        import comfy.utils
        from .sharp.gaussians import Gaussians3D, save_ply, unproject_gaussians

        # ---- unpack the multiband ---------------------------------------
        if not isinstance(gaussian_attrs, dict) or "samples" not in gaussian_attrs:
            raise ValueError(
                f"SharpImageAttrsToPLY: gaussian_attrs must be a MULTIBAND_IMAGE "
                f"dict with key 'samples', got {type(gaussian_attrs).__name__}"
            )
        samples = gaussian_attrs["samples"]  # [B, C, H, W]
        if samples.dim() != 4 or samples.shape[1] < 14:
            raise ValueError(
                f"SharpImageAttrsToPLY: samples must be [B, 14+, H, W], "
                f"got {tuple(samples.shape)}"
            )
        B, C, H, W = samples.shape
        if C < 14:
            raise ValueError(f"need at least 14 channels, got {C}")

        # ---- normalize camera inputs ------------------------------------
        # If extrinsics/intrinsics aren't wired (e.g. single-image debug
        # path), fall back to identity / focal-length-derived K so the
        # node still produces a usable PLY without forcing the user to
        # construct a CameraSample upstream.
        if extrinsics is None:
            ext = torch.eye(4).float().unsqueeze(0).repeat(B, 1, 1)
        else:
            ext = extrinsics.detach().float() if isinstance(extrinsics, torch.Tensor) \
                else torch.as_tensor(np.asarray(extrinsics, dtype=np.float32)).float()
            if ext.dim() == 2:
                ext = ext.unsqueeze(0)
            if ext.shape[0] != B:
                raise ValueError(
                    f"extrinsics batch {ext.shape[0]} != multiband batch {B}"
                )

        if intrinsics is None:
            f_px_default = float(focal_length_px) if focal_length_px > 0 \
                else max(W, H) / 2.0  # ~90° FOV fallback
            K_default = torch.tensor([
                [f_px_default, 0.0,          W / 2.0],
                [0.0,          f_px_default, H / 2.0],
                [0.0,          0.0,          1.0],
            ], dtype=torch.float32)
            intr = K_default.unsqueeze(0).repeat(B, 1, 1)
            _p(f"intrinsics not wired -> using identity-style K (fx=fy={f_px_default:.1f}, "
               f"cx={W/2:.1f}, cy={H/2:.1f}). Pass `intrinsics` from "
               f"SharpPredictGaussianAttrs for correct geometry.")
        else:
            intr = intrinsics.detach().float() if isinstance(intrinsics, torch.Tensor) \
                else torch.as_tensor(np.asarray(intrinsics, dtype=np.float32)).float()
            if intr.dim() == 2:
                intr = intr.unsqueeze(0)
            if intr.shape[0] != B:
                raise ValueError(
                    f"intrinsics batch {intr.shape[0]} != multiband batch {B}"
                )

        device = samples.device if samples.is_cuda else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        samples = samples.to(device).float()
        ext = ext.to(device)
        intr = intr.to(device)

        # ---- output path convention (matches SharpPredict) --------------
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        if B == 1:
            output_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_{timestamp}.ply")
            output_folder = None
            is_batch = False
        else:
            output_folder = os.path.join(OUTPUT_DIR, f"{output_prefix}_{timestamp}")
            os.makedirs(output_folder, exist_ok=True)
            output_path = output_folder
            is_batch = True

        all_ply_paths = []
        all_extrinsics = []
        all_intrinsics = []

        pbar = comfy.utils.ProgressBar(B)
        t_start = time.time()

        for b in range(B):
            # ---- extract per-pixel attrs -> Gaussians3D in NDC ------------
            s = samples[b]                       # [14, H, W]
            n_pixels = H * W

            # Multiband stores [C, H, W]; flatten spatial dims -> [C, N], then
            # transpose to [N, C] for the per-channel slices below.
            flat = s.reshape(C, n_pixels).T      # [N, C]
            mean_ndc = flat[:, POS].contiguous()          # [N, 3]
            scales   = flat[:, SCALE].contiguous()        # [N, 3]
            quats    = flat[:, QUAT].contiguous()         # [N, 4]
            colors   = flat[:, COLOR].contiguous()        # [N, 3]
            opacs    = flat[:, OPACITY].contiguous()      # [N]

            # Gaussians3D expects [B=1, N, C] for mean/scale/quat/color,
            # [B=1, N] for opacity.
            gaussians_ndc = Gaussians3D(
                mean_vectors=mean_ndc.unsqueeze(0),
                singular_values=scales.unsqueeze(0),
                quaternions=quats.unsqueeze(0),
                colors=colors.unsqueeze(0),
                opacities=opacs.unsqueeze(0),
            )

            # ---- camera matrices for unprojection ------------------------
            ext_b = ext[b]                       # [4, 4]
            K_b = intr[b].clone()                # [3, 3] or [4, 4]
            if K_b.shape == (3, 3):
                K4 = torch.eye(4, device=device, dtype=K_b.dtype)
                K4[:3, :3] = K_b
                K_b = K4

            # SHARP's internal convention: scale K to internal_shape
            # (the resolution used during decode). The multiband H/W
            # already match that internal grid (768 for the gaussian
            # decoder head), so no further rescale is needed — but
            # `unproject_gaussians` wants image_shape=(H, W) matching K's
            # native resolution. The intrinsics input is already pixel-K
            # for (H, W) (per the rescale done in SharpPredictGaussianAttrs).
            image_shape = (H, W)

            # ---- NDC -> world ---------------------------------------------
            gaussians_world = unproject_gaussians(
                gaussians_ndc, ext_b, K_b, image_shape,
            )

            # ---- save PLY ------------------------------------------------
            if is_batch:
                ply_path = os.path.join(output_folder, f"{b+1:03d}.ply")
            else:
                ply_path = output_path

            f_px = float(focal_length_px) if focal_length_px > 0 \
                else float(K_b[0, 0].item())
            _, metadata = save_ply(
                gaussians_world, f_px, image_shape, Path(ply_path),
            )

            all_ply_paths.append(ply_path)
            all_extrinsics.append(metadata["extrinsic"])
            all_intrinsics.append(metadata["intrinsic"])
            pbar.update(1)

        elapsed = time.time() - t_start
        loc = output_folder if is_batch else output_path
        total_n = B * H * W
        _p(
            f"{B} multiband(s) {H}×{W} -> {total_n/1e6:.2f}M gaussians; "
            f"saved to {loc}; {elapsed:.1f}s"
        )

        return io.NodeOutput(
            output_path, all_extrinsics[0], all_intrinsics[0],
        )


NODE_CLASS_MAPPINGS = {
    "SharpImageAttrsToPLY": SharpImageAttrsToPLY,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpImageAttrsToPLY": "SHARP Image + Attrs to PLY",
}
