"""SharpRayToPlanarDepth — convert per-face ray-distance depth to planar-Z.

Equirect depth panoramas (e.g. PanoramaDepthMerge output) store
ray-distance — Euclidean distance from the camera center along each
equirect pixel's ray direction. After splitting back into per-face
crops via PanoramaSplit / SharpPanoramaIcosahedronSplit, those faces
inherit the ray-distance convention.

SHARP's gaussian decoder + unprojection expect planar-Z (depth along
the face's optical axis). At the corner of a 90° face the two differ
by ~73%; feeding ray-distance where planar-Z is expected causes the
same world point to land at different 3D positions across faces.

This node sits between the depth source and the SHARP predict node,
applying the per-face cos-map:

    cos_map[u, v] = 1 / sqrt(((u-cx)/fx)² + ((v-cy)/fy)² + 1)
    planar_z      = ray_distance × cos_map

so downstream SHARP nodes can keep their default `depth_convention=
planar_z` and stay convention-pure.

Inputs:
  image       (IMAGE)      — per-face depth in ray-distance convention
  extrinsics  (EXTRINSICS) — pass-through (not used in the math)
  intrinsics  (INTRINSICS) — per-face K; needed to build the cos-map.
                             Accepts normalized K (PanoPack convention,
                             fx~=0.5) or pixel K (Sharp convention,
                             fx~=hundreds) — auto-detected and rescaled.

Outputs:
  image       (IMAGE)      — per-face depth in planar-Z convention
  extrinsics  (EXTRINSICS) — pass-through
  intrinsics  (INTRINSICS) — pass-through (in pixel-K convention if it
                             came in normalized, so all downstream nodes
                             see a consistent unit)
"""

from __future__ import annotations

import sys

import torch
from comfy_api.latest import io


def _p(msg: str) -> None:
    print(f"[SharpRayToPlanarDepth] {msg}", file=sys.stderr, flush=True)


class SharpRayToPlanarDepth(io.ComfyNode):
    """Convert per-face ray-distance depth to planar-Z via the cos-map."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharpRayToPlanarDepth",
            display_name="SHARP Ray-Distance -> Planar Depth",
            category="SHARP",
            description=(
                "Per-face depth convention converter. Multiplies "
                "ray-distance depth by the per-face cos-map "
                "(1 / sqrt(((u-cx)/fx)² + ((v-cy)/fy)² + 1)) to produce "
                "planar-Z depth that SHARP's gaussian decoder and "
                "unprojection expect natively. Place between a "
                "panorama-split depth source (PanoramaSplit on an equirect "
                "depth pano) and the SHARP predict node. The intrinsics "
                "passthrough is rescaled to pixel-K if it came in "
                "normalized (PanoPack's utils3d.intrinsics_from_fov "
                "convention), so all downstream nodes see consistent units."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Per-face depth in ray-distance convention "
                            "(e.g. the output of PanoramaSplit on a depth "
                            "panorama). Shape [B, H, W, 3] or [B, H, W].",
                ),
                io.Custom("INTRINSICS").Input(
                    "intrinsics",
                    tooltip="Per-face camera intrinsics. Accepts either "
                            "normalized K (fx~=0.5 for 90° fov, PanoPack "
                            "convention) or pixel K (fx in the hundreds, "
                            "Sharp convention) — auto-detected. The "
                            "output intrinsics socket emits pixel-K so "
                            "all downstream consumers see the same units.",
                ),
                io.Custom("EXTRINSICS").Input(
                    "extrinsics", optional=True,
                    tooltip="Pass-through to the extrinsics output. Not "
                            "used in the cos-map computation (purely "
                            "intrinsic-dependent).",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="Per-face depth in planar-Z convention. Same "
                            "shape as the input depth.",
                ),
                io.Custom("EXTRINSICS").Output(
                    display_name="extrinsics",
                    tooltip="Pass-through of the input extrinsics.",
                ),
                io.Custom("INTRINSICS").Output(
                    display_name="intrinsics",
                    tooltip="Per-face camera intrinsics, rescaled to "
                            "pixel-K if the input was normalized.",
                ),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        image: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
    ):
        depth = image
        if depth.dim() == 3:
            depth = depth.unsqueeze(0)
        # IMAGE channel layout: [B, H, W, C] with C in {1, 3, 4}; depth is
        # broadcast across channels so we operate on a single channel and
        # then broadcast back at the end.
        if depth.dim() == 4 and depth.shape[-1] in (1, 3, 4):
            depth_2d_batch = depth[..., 0]            # [B, H, W]
            had_channel = True
            n_channels = int(depth.shape[-1])
        elif depth.dim() == 4 and depth.shape[1] in (1, 3, 4):
            depth_2d_batch = depth[:, 0]              # [B, H, W]  (legacy [B, C, H, W])
            had_channel = True
            n_channels = int(depth.shape[1])
        else:
            depth_2d_batch = depth                    # already [B, H, W]
            had_channel = False
            n_channels = 1
        B, H, W = depth_2d_batch.shape

        # Convention sniff: normalized K (PanoPack) vs pixel K (Sharp).
        intr = intrinsics
        intr_was_3d = intr.dim() == 3
        sample_fx = float(intr[0, 0, 0] if intr_was_3d else intr[0, 0])
        rescaled = False
        if sample_fx < 2.0:
            intr = intr.clone().float()
            if intr_was_3d:
                intr[:, 0, :] *= float(W)
                intr[:, 1, :] *= float(H)
            else:
                intr[0, :] *= float(W)
                intr[1, :] *= float(H)
            rescaled = True
            sample_fx_after = float(intr[0, 0, 0] if intr_was_3d else intr[0, 0])
            _p(f"detected normalized intrinsics (fx<2); rescaled to "
               f"pixel-K for {W}×{H}: fx={sample_fx_after:.1f}")

        # Apply per-face cos-map. K may be 2-D (single matrix shared across
        # the batch) or 3-D (per-face). Build a per-batch cos_map either way.
        device = depth_2d_batch.device
        uu = torch.arange(W, dtype=torch.float32, device=device)
        vv = torch.arange(H, dtype=torch.float32, device=device)
        uu_g, vv_g = torch.meshgrid(uu, vv, indexing="xy")  # (H, W) each

        if intr_was_3d:
            if intr.shape[0] != B:
                raise ValueError(
                    f"SharpRayToPlanarDepth: intrinsics batch {intr.shape[0]} "
                    f"!= depth batch {B}"
                )
            out_batch = []
            for b in range(B):
                fx = float(intr[b, 0, 0]); fy = float(intr[b, 1, 1])
                cx = float(intr[b, 0, 2]); cy = float(intr[b, 1, 2])
                x_cam = (uu_g - cx) / fx
                y_cam = (vv_g - cy) / fy
                cos_map = 1.0 / torch.sqrt(x_cam * x_cam + y_cam * y_cam + 1.0)
                out_batch.append(depth_2d_batch[b] * cos_map)
            planar = torch.stack(out_batch, dim=0)
            cos_min_log = "per-face (3D K)"
        else:
            fx = float(intr[0, 0]); fy = float(intr[1, 1])
            cx = float(intr[0, 2]); cy = float(intr[1, 2])
            x_cam = (uu_g - cx) / fx
            y_cam = (vv_g - cy) / fy
            cos_map = 1.0 / torch.sqrt(x_cam * x_cam + y_cam * y_cam + 1.0)
            planar = depth_2d_batch * cos_map.unsqueeze(0)
            cos_min_log = f"min={float(cos_map.min()):.4f} max={float(cos_map.max()):.4f}"

        _p(f"ray->planar: B={B} {H}×{W}, cos_map {cos_min_log}, "
           f"intrinsics{'_rescaled' if rescaled else ''} fx={fx:.1f} cx={cx:.1f}")

        # Restore channel dim if input had one.
        if had_channel:
            planar = planar.unsqueeze(-1).expand(-1, -1, -1, n_channels).contiguous()

        return io.NodeOutput(planar, extrinsics, intr)


NODE_CLASS_MAPPINGS = {"SharpRayToPlanarDepth": SharpRayToPlanarDepth}
NODE_DISPLAY_NAME_MAPPINGS = {"SharpRayToPlanarDepth": "SHARP Ray-Distance -> Planar Depth"}
