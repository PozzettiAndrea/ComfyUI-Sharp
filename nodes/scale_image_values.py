"""ScaleImageValues — linearly remap image values to [0, 1].

Use when an IMAGE socket carries values outside [0, 1] (e.g.
SharpDepthMerge.depth in meters, SharpPredictMetricDepth.layer_*_metric_depth,
SharpPredictGaussianAttrs.metric_depth) and you want to feed it into a
ComfyUI PreviewImage / SaveImage that expects [0, 1] RGB.

Computes per-image min and max across all pixels and all channels, then
linearly remaps: `(x - min) / (max - min)`. Result is in [0, 1] (clamped
defensively). Each item in the batch is scaled independently — no
information leaks between batch elements.
"""

import sys

import torch

from comfy_api.latest import io


def _p(msg: str) -> None:
    print(f"[ScaleImageValues] {msg}", file=sys.stderr, flush=True)


class ScaleImageValues(io.ComfyNode):
    """Linearly scale image values to the [0, 1] range using min/max stretch."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ScaleImageValues",
            display_name="Scale Image Values",
            category="SHARP",
            description=(
                "Linearly remap an IMAGE's values to [0, 1]. Useful for "
                "previewing depth maps (which can have values in meters, "
                "outside the [0, 1] range ComfyUI's PreviewImage expects). "
                "Each batch item is scaled independently using its own "
                "min/max across all (H, W, C). Constant-valued images "
                "become all-zeros (rather than NaN from divide-by-zero)."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input IMAGE [B, H, W, 3] (or [H, W, 3] for "
                            "single image). Any value range — depth in "
                            "meters, HDR linear RGB, arbitrary scalars "
                            "broadcast across 3 channels, etc."),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="Same shape as input, values linearly remapped "
                            "to [0, 1] per-batch-item."),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        B = image.shape[0]

        # Per-batch-item min/max across all (H, W, C).
        flat = image.view(B, -1).float()
        img_min = flat.min(dim=1, keepdim=True).values   # [B, 1]
        img_max = flat.max(dim=1, keepdim=True).values   # [B, 1]
        denom = (img_max - img_min).clamp(min=1e-6)
        scaled_flat = (flat - img_min) / denom
        scaled = scaled_flat.view_as(image).clamp(0.0, 1.0)

        _p(
            f"{B} image(s) {tuple(image.shape[1:])}: "
            f"in [{float(img_min.min()):.4f}, {float(img_max.max()):.4f}] "
            f"-> out [0, 1]"
        )
        return io.NodeOutput(scaled)


NODE_CLASS_MAPPINGS = {
    "ScaleImageValues": ScaleImageValues,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScaleImageValues": "Scale Image Values",
}
