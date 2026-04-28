"""MergeGaussians node for ComfyUI-Sharp.

Merges multiple PLY files (from panorama samples) into a single unified Gaussian scene.
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from comfy_api.latest import io

log = logging.getLogger("sharp")


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
                io.String.Output(display_name="ply_path"),
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

        # Find all PLY files
        ply_folder = Path(ply_folder)
        if not ply_folder.exists():
            raise ValueError(f"PLY folder does not exist: {ply_folder}")

        ply_files = sorted(ply_folder.glob("*.ply"))
        if not ply_files:
            raise ValueError(f"No PLY files found in: {ply_folder}")

        log.info(f"Found {len(ply_files)} PLY files to merge")

        # Load all PLY files
        all_positions = []
        all_colors = []
        all_scales = []
        all_rotations = []
        all_opacities = []

        for i, ply_path in enumerate(ply_files):
            log.info(f"Loading {ply_path.name} ({i+1}/{len(ply_files)})")
            data = load_ply_simple(str(ply_path))

            positions = data['positions']
            colors = data['colors']
            scales = data['scales']
            rotations = data['rotations']
            opacities = data['opacities']

            # Apply filters
            mask = np.ones(len(positions), dtype=bool)

            if max_depth > 0:
                # Filter by depth (distance from origin)
                depths = np.linalg.norm(positions, axis=-1)
                mask &= depths <= max_depth

            if min_opacity > 0:
                mask &= opacities >= min_opacity

            filtered_count = (~mask).sum()
            if filtered_count > 0:
                log.info(f"  Filtered out {filtered_count:,} Gaussians")

            all_positions.append(positions[mask])
            all_colors.append(colors[mask])
            all_scales.append(scales[mask])
            all_rotations.append(rotations[mask])
            all_opacities.append(opacities[mask])

        # Concatenate all
        merged_positions = np.concatenate(all_positions, axis=0)
        merged_colors = np.concatenate(all_colors, axis=0)
        merged_scales = np.concatenate(all_scales, axis=0)
        merged_rotations = np.concatenate(all_rotations, axis=0)
        merged_opacities = np.concatenate(all_opacities, axis=0)

        num_gaussians = len(merged_positions)
        log.info(f"Total Gaussians after merge: {num_gaussians:,}")

        # Save merged PLY
        timestamp = int(time.time() * 1000)
        output_filename = f"{output_prefix}_{timestamp}.ply"
        output_path = ply_folder.parent / output_filename

        log.info(f"Saving to {output_path}")
        save_merged_ply(
            merged_positions,
            merged_colors,
            merged_scales,
            merged_rotations,
            merged_opacities,
            str(output_path),
        )

        log.info(f"Done! Merged {len(ply_files)} files into {num_gaussians:,} Gaussians")

        return io.NodeOutput(str(output_path), num_gaussians)


NODE_CLASS_MAPPINGS = {
    "MergeGaussians": MergeGaussians,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MergeGaussians": "Merge Gaussians (PLY Files)",
}
