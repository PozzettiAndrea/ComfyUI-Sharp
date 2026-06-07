"""MergeGaussians node for ComfyUI-Sharp.

Merges multiple PLY files (from panorama samples) into a single unified Gaussian scene.
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


def _p(msg: str) -> None:
    """Print to stderr so the line shows up in ComfyUI's worker log."""
    print(f"[MergeGaussians] {msg}", file=sys.stderr, flush=True)


def load_ply_simple(path: str) -> dict:
    """Load Gaussian data from PLY file.

    Returns dict with arrays: positions, colors, scales, rotations, opacities
    """
    from plyfile import PlyData
    plydata = PlyData.read(path)
    vertex = plydata['vertex']

    positions = np.stack([
        vertex['x'],
        vertex['y'],
        vertex['z']
    ], axis=-1)

    # Colors (SH coefficients - we take DC term)
    colors = np.stack([
        vertex['f_dc_0'],
        vertex['f_dc_1'],
        vertex['f_dc_2']
    ], axis=-1)

    # Scales
    scales = np.stack([
        vertex['scale_0'],
        vertex['scale_1'],
        vertex['scale_2']
    ], axis=-1)

    # Rotations (quaternion)
    rotations = np.stack([
        vertex['rot_0'],
        vertex['rot_1'],
        vertex['rot_2'],
        vertex['rot_3']
    ], axis=-1)

    # Opacity
    opacities = vertex['opacity']

    return {
        'positions': positions,
        'colors': colors,
        'scales': scales,
        'rotations': rotations,
        'opacities': opacities,
    }


def save_field_ply(
    positions: np.ndarray,
    face_ids: np.ndarray,
    output_path: str,
) -> None:
    """Save a minimal PLY with just (x, y, z) + a `face_id` int32 scalar
    field per vertex. Used for visualization / per-view-origin analysis
    (color by face_id in ParaView / VTK / Three.js etc.). No SH, colors,
    scales, rotations, or opacities — pure point cloud with provenance."""
    n = int(positions.shape[0])
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('face_id', 'i4'),
    ]
    elements = np.empty(n, dtype=dtype)
    elements['x'] = positions[:, 0]
    elements['y'] = positions[:, 1]
    elements['z'] = positions[:, 2]
    elements['face_id'] = face_ids.astype(np.int32)

    from plyfile import PlyData, PlyElement
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)


def save_merged_ply(
    positions: np.ndarray,
    colors: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    opacities: np.ndarray,
    output_path: str,
):
    """Save merged Gaussians to PLY file."""

    num_gaussians = len(positions)

    # Create structured array
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]

    elements = np.empty(num_gaussians, dtype=dtype)
    elements['x'] = positions[:, 0]
    elements['y'] = positions[:, 1]
    elements['z'] = positions[:, 2]
    elements['f_dc_0'] = colors[:, 0]
    elements['f_dc_1'] = colors[:, 1]
    elements['f_dc_2'] = colors[:, 2]
    elements['opacity'] = opacities
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    elements['rot_0'] = rotations[:, 0]
    elements['rot_1'] = rotations[:, 1]
    elements['rot_2'] = rotations[:, 2]
    elements['rot_3'] = rotations[:, 3]

    from plyfile import PlyData, PlyElement
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)


class MergeGaussians(io.ComfyNode):
    """Merge multiple Gaussian PLY files into a single scene.

    Used after running SHARP on panorama samples to combine all views.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MergeGaussians",
            display_name="Merge Gaussians (PLY Files)",
            category="SHARP",
            description="Merge multiple Gaussian PLY files into a single unified scene.",
            is_output_node=True,
            inputs=[
                io.String.Input("ply_folder",
                                tooltip="Path to folder containing PLY files to merge"),
                io.String.Input("output_prefix", default="merged", optional=True,
                                tooltip="Prefix for output merged PLY file"),
                io.Float.Input("max_depth", default=0.0, min=0.0, max=1000.0, step=0.1,
                               optional=True,
                               tooltip="Filter out Gaussians beyond this depth (0 = no filter)"),
                io.Float.Input("min_opacity", default=0.0, min=0.0, max=1.0, step=0.01,
                               optional=True,
                               tooltip="Filter out Gaussians with opacity below this (0 = no filter)"),
            ],
            outputs=[
                io.String.Output(
                    display_name="ply_paths",
                    tooltip="Passthrough of the input ply_folder. Wire to "
                            "downstream nodes that want to operate on the "
                            "original per-face PLYs (e.g. SharpDedupGaussiansEquirect "
                            "on the merged_ply_path while still keeping access "
                            "to the originals)."),
                io.String.Output(
                    display_name="merged_ply_path",
                    tooltip="Path to the merged output PLY file (a single "
                            "PLY containing every gaussian from every input "
                            "PLY, post-filter)."),
                io.String.Output(
                    display_name="field_ply_path",
                    tooltip="Path to a minimal PLY with just (x, y, z) + a "
                            "`face_id` int32 scalar field per vertex. Same "
                            "vertex count and ordering as `merged_ply_path` "
                            "but no SH/colors/scales/rotations/opacities. "
                            "Useful for ParaView / VTK / Three.js to color "
                            "points by which input PLY they came from — "
                            "shows directly which views contribute where, "
                            "and how much overlap exists between views."),
                io.Int.Output(display_name="num_gaussians"),
            ],
        )

    @classmethod
    def execute(
        cls,
        ply_folder: str,
        output_prefix: str = "merged",
        max_depth: float = 0.0,
        min_opacity: float = 0.0,
    ):
        """Merge all PLY files in folder."""
        import comfy.utils

        t_start = time.time()

        ply_folder_in = ply_folder
        ply_folder = Path(ply_folder)
        if not ply_folder.exists():
            raise ValueError(f"PLY folder does not exist: {ply_folder_in!r}")

        ply_files = sorted(ply_folder.glob("*.ply"))
        if not ply_files:
            raise ValueError(f"No PLY files found in: {ply_folder}")

        filt_str = []
        if max_depth > 0:
            filt_str.append(f"max_depth={max_depth:.2f}m")
        if min_opacity > 0:
            filt_str.append(f"min_opacity={min_opacity:.3f}")
        filt_msg = (" + filters " + ", ".join(filt_str)) if filt_str else " (no filters)"
        _p(f"merging {len(ply_files)} PLY file(s) from {ply_folder}{filt_msg}")

        all_positions = []
        all_colors = []
        all_scales = []
        all_rotations = []
        all_opacities = []
        all_face_ids = []   # per-vertex source-PLY index, matches merged ordering
        n_in_total = 0
        n_filtered_total = 0

        pbar = comfy.utils.ProgressBar(len(ply_files))
        for i, ply_path in enumerate(ply_files):
            data = load_ply_simple(str(ply_path))
            positions = data['positions']
            colors = data['colors']
            scales = data['scales']
            rotations = data['rotations']
            opacities = data['opacities']

            n_before = len(positions)
            n_in_total += n_before

            mask = np.ones(n_before, dtype=bool)
            if max_depth > 0:
                depths = np.linalg.norm(positions, axis=-1)
                mask &= depths <= max_depth
            if min_opacity > 0:
                mask &= opacities >= min_opacity
            filtered_count = int((~mask).sum())
            n_filtered_total += filtered_count

            kept = int(mask.sum())
            all_positions.append(positions[mask])
            all_colors.append(colors[mask])
            all_scales.append(scales[mask])
            all_rotations.append(rotations[mask])
            all_opacities.append(opacities[mask])
            # Per-vertex source-PLY index (i = position of this file in
            # the sorted ply_files list). face_id assignments survive the
            # filter mask because we only append the kept count.
            all_face_ids.append(np.full(kept, i, dtype=np.int32))
            pbar.update(1)

        merged_positions = np.concatenate(all_positions, axis=0)
        merged_colors = np.concatenate(all_colors, axis=0)
        merged_scales = np.concatenate(all_scales, axis=0)
        merged_rotations = np.concatenate(all_rotations, axis=0)
        merged_opacities = np.concatenate(all_opacities, axis=0)
        merged_face_ids = np.concatenate(all_face_ids, axis=0)
        num_gaussians = len(merged_positions)

        if n_filtered_total > 0:
            pct = 100.0 * n_filtered_total / max(n_in_total, 1)
            _p(f"filtered {n_filtered_total:,} / {n_in_total:,} gaussians ({pct:.1f}%)")

        timestamp = int(time.time() * 1000)
        output_filename = f"{output_prefix}_{timestamp}.ply"
        output_path = ply_folder.parent / output_filename
        field_output_filename = f"{output_prefix}_{timestamp}_fields.ply"
        field_output_path = ply_folder.parent / field_output_filename

        t_save = time.time()
        save_merged_ply(
            merged_positions,
            merged_colors,
            merged_scales,
            merged_rotations,
            merged_opacities,
            str(output_path),
        )
        save_field_ply(
            merged_positions,
            merged_face_ids,
            str(field_output_path),
        )
        save_elapsed = time.time() - t_save

        try:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            size_str = f", {size_mb:.1f} MB"
        except Exception:
            size_str = ""

        elapsed = time.time() - t_start
        _p(
            f"merged {len(ply_files)} file(s) -> {num_gaussians/1e6:.2f}M gaussians "
            f"-> {output_path}{size_str}; "
            f"save {save_elapsed:.1f}s, total {elapsed:.1f}s"
        )
        _p(f"field PLY (positions + face_id only) -> {field_output_path}")

        return io.NodeOutput(
            str(ply_folder_in), str(output_path),
            str(field_output_path), num_gaussians,
        )


NODE_CLASS_MAPPINGS = {
    "MergeGaussians": MergeGaussians,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MergeGaussians": "Merge Gaussians (PLY Files)",
}
