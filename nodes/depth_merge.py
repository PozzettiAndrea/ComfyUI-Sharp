"""SharpDepthMerge node for ComfyUI-Sharp.

Per-face rectilinear depth maps -> a single equirectangular distance map,
via MoGe's sparse-LSMR formulation in log-distance space (Laplacian +
gradient terms). 1:1 schema mirror of WorldNavDepthMerge — wiring is
identical so workflows port over by editing only the node-class name.

Backed by the same solver as WorldNav: vendored at
`nodes/_vendor/moge_panorama.py` (CPU LSMR via scipy) + vendored at
`nodes/_vendor/worldgen/src/panorama_utils.py` (GPU LSMR via
solve_lsmr_gpu + post-merge sky inpaint + south-pole smooth).
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import torch
from comfy_api.latest import io

log = logging.getLogger("sharp")


def _p(msg: str) -> None:
    print(f"[SharpDepthMerge] {msg}", file=sys.stderr, flush=True)


class SharpDepthMerge(io.ComfyNode):
    """Per-face depth maps + per-face extrinsics/intrinsics -> equirect depth IMAGE.

    Schema is a 1:1 mirror of WorldNavDepthMerge so workflows port over
    without rewiring. Internally uses the same vendored MoGe LSMR solver.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharpDepthMerge",
            display_name="Sharp Depth Merge",
            category="SHARP",
            description=(
                "Stitch per-face rectilinear depth maps from a panorama "
                "split (SharpPanoramaIcosahedronSplit etc.) back into a "
                "single equirect distance map. Sparse-LSMR formulation in "
                "log-distance space (Laplacian + gradient terms in MoGe "
                "style). 1:1 schema mirror of WorldNavDepthMerge."
            ),
            inputs=[
                io.Image.Input(
                    "face_depths",
                    tooltip="Per-face depth maps from a rectilinear depth "
                            "model (MoGe2 / SHARP / etc.). Shape "
                            "(N, h, w, C); only the first channel is used "
                            "as the distance value."),
                io.Mask.Input(
                    "face_valid_masks",
                    optional=True,
                    tooltip="Per-face valid masks (typically the depth "
                            "model's valid_mask output). Shape (N, h, w) "
                            "bool/float. Load-bearing: upstream "
                            "merge_panorama_depth uses this to gate which "
                            "pixels enter the gradient + laplacian LSMR "
                            "system. Without it, sky / invalid pixels "
                            "corrupt the solve. If unwired, all pixels "
                            "are treated as valid (degraded fallback)."),
                io.Custom("EXTRINSICS").Input(
                    "extrinsics",
                    tooltip="From the panorama-split node. (N, 4, 4) "
                            "world-to-camera per face."),
                io.Custom("INTRINSICS").Input(
                    "intrinsics",
                    tooltip="From the panorama-split node. (N, 3, 3) "
                            "pixel-space per face."),
                io.Int.Input(
                    "out_width", default=1920, min=512, max=4096, step=64,
                    tooltip="Output equirect width. 2:1 aspect (height = "
                            "width / 2 is the convention; provide "
                            "out_height to override)."),
                io.Int.Input(
                    "out_height", default=960, min=256, max=2048, step=64,
                    tooltip="Output equirect height. Should be width/2 "
                            "for 2:1 aspect."),
                io.Boolean.Input(
                    "use_gpu", default=True,
                    tooltip="DEPRECATED. Kept for back-compat. Equivalent "
                            "to strategy='cpu_fallback' when False. "
                            "Otherwise respects strategy."),
                io.Combo.Input(
                    "strategy",
                    options=["auto", "full", "vcycle", "cpu_fallback"],
                    default="auto",
                    tooltip=(
                        "Solver strategy for the equirect depth output:\n"
                        "  auto: full if out<=1920x960 else vcycle (recommended).\n"
                        "  full: solve LSMR at out_resolution (original). OOMs above ~2 Mpix on 24 GB.\n"
                        "  vcycle: solve LSMR at 1920x960 then bilinear-upsample depth+mask "
                        "to out_resolution. ~4x faster + fits memory at 3840x1920. Per-face "
                        "info saturates at 1920x960 so quality is indistinguishable.\n"
                        "  cpu_fallback: scipy LSMR on CPU end-to-end. Slow (~30-60s at "
                        "1920x960) but works at any resolution."
                    )),
            ],
            outputs=[
                io.Image.Output(display_name="depth"),
                io.Mask.Output(display_name="valid_mask"),
            ],
        )

    @classmethod
    def execute(cls, face_depths, extrinsics, intrinsics,
                face_valid_masks=None, out_width=1920, out_height=960,
                use_gpu=True, strategy="auto"):
        import cv2
        # Vendored MoGe LSMR solvers — imported here so depth_merge.py
        # loads cleanly even if scipy / utils3d / etc. aren't installed.
        # (They ARE in Sharp's pixi env per comfy-env.toml; this lazy
        # import keeps the node-class registration unconditional.)
        from ._vendor.moge_panorama import merge_panorama_depth
        from ._vendor.worldgen.src.panorama_utils import (
            merge_panorama_depth_gpu,
            smooth_south_pole_depth,
        )

        # --- face_depths: (N, h, w, C) → list of (h, w) float ---
        d = (face_depths.detach().cpu().numpy()
             if isinstance(face_depths, torch.Tensor) else np.asarray(face_depths))
        if d.ndim != 4:
            raise ValueError(
                f"SharpDepthMerge: face_depths must be (N, h, w, C), got {d.shape}"
            )
        N, fh, fw, _ = d.shape
        distance_maps = [d[i, ..., 0].astype(np.float32) for i in range(N)]

        # --- extrinsics / intrinsics: (N, 4, 4) and (N, 3, 3) ---
        ex = (extrinsics.detach().cpu().numpy()
              if isinstance(extrinsics, torch.Tensor) else np.asarray(extrinsics))
        intr = (intrinsics.detach().cpu().numpy()
                if isinstance(intrinsics, torch.Tensor) else np.asarray(intrinsics))
        if ex.shape != (N, 4, 4) or intr.shape != (N, 3, 3):
            raise ValueError(
                f"SharpDepthMerge: shape mismatch — extrinsics {ex.shape}, "
                f"intrinsics {intr.shape}, expected ({N},4,4) and ({N},3,3)"
            )
        extr_list = [ex[i].astype(np.float32) for i in range(N)]
        intr_list = [intr[i].astype(np.float32) for i in range(N)]

        # --- valid_masks: (N, h, w, C) or None → list of (h, w) bool ---
        if face_valid_masks is None:
            pred_masks = [np.ones((fh, fw), dtype=bool) for _ in range(N)]
        else:
            v = (face_valid_masks.detach().cpu().numpy()
                 if isinstance(face_valid_masks, torch.Tensor)
                 else np.asarray(face_valid_masks))
            if v.ndim == 4:
                v = v[..., 0]
            if v.shape != (N, fh, fw):
                raise ValueError(
                    f"SharpDepthMerge: face_valid_masks shape {v.shape} "
                    f"doesn't match face_depths ({N}, {fh}, {fw})"
                )
            pred_masks = [(v[i] > 0.5).astype(bool) for i in range(N)]

        _p(f"merging {N} faces @ {fh}x{fw} -> equirect {out_height}x{out_width} "
           f"(use_gpu={use_gpu})")

        # Progress bar — visible in ComfyUI if available.
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(100)
        except Exception:
            class _Noop:
                def update_absolute(self, *_a, **_k):
                    pass
            pbar = _Noop()
        pbar.update_absolute(1, 100)

        # Resolve strategy. use_gpu=False overrides → cpu_fallback (back-compat
        # for the equivalent toggle in WorldNavDepthMerge).
        out_w, out_h = int(out_width), int(out_height)
        resolved = strategy
        if not use_gpu:
            resolved = "cpu_fallback"
        if resolved == "auto":
            resolved = "vcycle" if (out_w * out_h > 1920 * 960) else "full"
        _p(f"resolved strategy={resolved} (requested={strategy}, "
           f"use_gpu={use_gpu}, out={out_w}x{out_h})")

        if resolved == "full":
            # Recursive LSMR all the way up to out_resolution. Reference;
            # OOMs above ~2 Mpix on 24 GB cards.
            depth_np, mask_np = merge_panorama_depth_gpu(
                out_w, out_h,
                distance_maps, pred_masks,
                extr_list, intr_list,
            )
        elif resolved == "vcycle":
            # V-cycle: solve the recursive pyramid up to 1920x960 (the
            # information-content ceiling for typical 42x ~512² per-face
            # data), then bilinear-upsample depth + nearest-upsample mask
            # to out_resolution. Skips the OOM-prone outermost LSMR.
            depth_low, mask_low = merge_panorama_depth_gpu(
                1920, 960,
                distance_maps, pred_masks,
                extr_list, intr_list,
            )
            if (out_w, out_h) == (1920, 960):
                depth_np, mask_np = depth_low, mask_low
            else:
                depth_np = cv2.resize(
                    depth_low.astype(np.float32),
                    (out_w, out_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask_np = cv2.resize(
                    mask_low.astype(np.uint8),
                    (out_w, out_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                _p(f"vcycle: upsampled 1920x960 -> {out_w}x{out_h} "
                   f"(bilinear depth, nearest mask)")
        elif resolved == "cpu_fallback":
            # scipy LSMR end-to-end on CPU. Slow but no VRAM ceiling.
            depth_np, mask_np = merge_panorama_depth(
                out_w, out_h,
                distance_maps, pred_masks,
                extr_list, intr_list,
                pbar=pbar,
            )
        else:
            raise ValueError(f"SharpDepthMerge: unknown strategy {resolved!r}")

        # Match upstream HY-World 2.0 post-merge processing
        # (panorama_utils.py:572-580). Two steps:
        #   1. Inpaint invalid pixels with the 99.9th-percentile (sky)
        #      depth instead of zero. Zero-depth puts those pixels right
        #      on the camera origin and breaks downstream point-cloud /
        #      mesh geometry.
        #   2. Smooth the south-pole strip (bottom 5%) to fix left-right
        #      seams the LSMR solver can't enforce due to the equirect
        #      parameterization stretching infinitely at the pole.
        depth_np = depth_np.astype(np.float32)
        valid = mask_np.astype(bool)
        if valid.any() and (~valid).any():
            sky_depth = float(np.nanquantile(depth_np[valid], 0.999))
            depth_np[~valid] = sky_depth
            _p(f"  inpainted {int((~valid).sum())} invalid pixels with "
               f"sky_depth={sky_depth:.3f}")
        depth_np = smooth_south_pole_depth(depth_np, smooth_height_ratio=0.05)
        # Belt-and-braces: trap any residual non-finite values that escaped.
        depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
        _p(f"merged depth: shape {depth_np.shape}, min={float(depth_np.min()):.3f}, "
           f"median={float(np.median(depth_np)):.3f}, max={float(depth_np.max()):.3f}; "
           f"valid pixels: {int(mask_np.sum())} / {mask_np.size}")

        # IMAGE convention: (B, H, W, C). Broadcast depth across 3 channels so
        # it composes with regular depth-viz nodes.
        depth_img = (
            torch.from_numpy(depth_np)
            .unsqueeze(-1).expand(-1, -1, 3).unsqueeze(0).contiguous()
        )
        valid_mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)

        pbar.update_absolute(100, 100)
        return io.NodeOutput(depth_img, valid_mask)


NODE_CLASS_MAPPINGS = {
    "SharpDepthMerge": SharpDepthMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpDepthMerge": "Sharp Depth Merge",
}
