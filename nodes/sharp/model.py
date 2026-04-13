"""All neural network modules for SHARP monocular 3D Gaussian Splatting predictor.

Consolidated from ~40 files into a single ComfyUI-native module.
All learnable layers use operations= for ComfyUI weight management.
Custom ViT replaces timm dependency with matching key names for checkpoint compatibility.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import abc
import copy
import logging
import math
from typing import Literal, NamedTuple, Sequence, Tuple, Union

log = logging.getLogger("sharp")

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
from comfy.ldm.modules.attention import optimized_attention_for_device

from .color_space import ColorSpace, sRGB2linearRGB
from .gaussians import Gaussians3D

# Activation helpers (replaces math.py abstraction — only sigmoid/exp used at inference)
def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))

def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.exp(x) - 1)

_ACTIVATIONS = {
    "sigmoid": (torch.sigmoid, _inverse_sigmoid),
    "exp": (torch.exp, torch.log),
}

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

NormLayerName = Literal["noop", "batch_norm", "group_norm", "instance_norm"]
UpsamplingMode = Literal["transposed_conv", "nearest", "bilinear"]
DimsDecoder = Tuple[int, int, int, int, int]
DPTImageEncoderType = Literal["skip_conv", "skip_conv_kernel2"]
ColorInitOption = Literal["none", "first_layer", "all_layers"]
DepthInitOption = Literal["surface_min", "surface_max", "base_depth", "linear_disparity"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def norm_layer_2d(
    num_features: int,
    norm_type: NormLayerName,
    num_groups: int = 8,
    dtype=None,
    device=None,
    operations=None,
) -> nn.Module:
    """Create normalization layer."""
    if operations is None:
        operations = comfy.ops.manual_cast
    if norm_type == "noop":
        return nn.Identity()
    elif norm_type == "batch_norm":
        return nn.BatchNorm2d(num_features=num_features)
    elif norm_type == "group_norm":
        return operations.GroupNorm(
            num_channels=num_features, num_groups=num_groups, dtype=dtype, device=device
        )
    elif norm_type == "instance_norm":
        return nn.InstanceNorm2d(num_features=num_features)
    else:
        raise ValueError(f"Invalid normalization layer type: {norm_type}")


def upsampling_layer(
    upsampling_mode: UpsamplingMode,
    scale_factor: int,
    dim_in: int,
    dtype=None,
    device=None,
    operations=None,
) -> nn.Module:
    """Create upsampling layer."""
    if operations is None:
        operations = comfy.ops.manual_cast
    if upsampling_mode == "transposed_conv":
        return operations.ConvTranspose2d(
            in_channels=dim_in,
            out_channels=dim_in,
            kernel_size=scale_factor,
            stride=scale_factor,
            padding=0,
            bias=False,
            dtype=dtype,
            device=device,
        )
    elif upsampling_mode in ("nearest", "bilinear"):
        return nn.Upsample(scale_factor=scale_factor, mode=upsampling_mode)
    else:
        raise ValueError(f"Invalid upsampling mode {upsampling_mode}.")


# ---------------------------------------------------------------------------
# NamedTuples
# ---------------------------------------------------------------------------


class GaussianBaseValues(NamedTuple):
    """Base values for gaussian predictor."""

    mean_x_ndc: torch.Tensor
    mean_y_ndc: torch.Tensor
    mean_inverse_z_ndc: torch.Tensor
    scales: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor


class InitializerOutput(NamedTuple):
    """Output of initializer."""

    gaussian_base_values: GaussianBaseValues
    feature_input: torch.Tensor
    global_scale: torch.Tensor | None = None


class MonodepthOutput(NamedTuple):
    """Output of the monodepth model."""

    disparity: torch.Tensor
    encoder_features: list[torch.Tensor]
    decoder_features: torch.Tensor
    output_features: list[torch.Tensor]
    intermediate_features: list[torch.Tensor] = []


class ImageFeatures(NamedTuple):
    """Image feature extracted from decoder."""

    texture_features: torch.Tensor
    geometry_features: torch.Tensor


# ===========================================================================
# Vision Transformer (replaces timm dependency)
# Attribute names match timm exactly for checkpoint key compatibility.
# ===========================================================================


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding. Matches timm PatchEmbed key naming."""

    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # Key: patch_embed.proj.weight / .bias
        self.proj = operations.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
            dtype=dtype, device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, N, C]
        return x


class LayerScale(nn.Module):
    """Per-channel learnable scaling. Key: ls{1,2}.gamma"""

    def __init__(self, dim: int, init_values: float = 1e-5, dtype=None, device=None):
        super().__init__()
        self.gamma = nn.Parameter(
            torch.full((dim,), init_values, dtype=dtype, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.to(device=x.device, dtype=x.dtype)


class ViTAttention(nn.Module):
    """ViT self-attention with fused qkv and ComfyUI optimized_attention.

    Keys: attn.qkv.{weight,bias}, attn.proj.{weight,bias}
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        self.num_heads = num_heads
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, transformer_options={}) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)  # Each [B, N, C]
        out = optimized_attention_for_device(q.device)(q, k, v, heads=self.num_heads,
                                                     transformer_options=transformer_options)
        return self.proj(out)


class ViTMlp(nn.Module):
    """ViT MLP supporting vanilla (GELU) and GLU (GEGLU) modes.

    Keys: mlp.fc1.{weight,bias}, mlp.fc2.{weight,bias}
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        use_glu: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        hidden = int(dim * mlp_ratio)
        self.use_glu = use_glu
        if use_glu:
            # GluMlp: fc1 outputs hidden (chunked into 2 halves for gating)
            self.fc1 = operations.Linear(dim, hidden, dtype=dtype, device=device)
            self.fc2 = operations.Linear(hidden // 2, dim, dtype=dtype, device=device)
        else:
            self.fc1 = operations.Linear(dim, hidden, dtype=dtype, device=device)
            self.fc2 = operations.Linear(hidden, dim, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.use_glu:
            x1, x2 = x.chunk(2, dim=-1)
            x = x1 * F.gelu(x2)
        else:
            x = F.gelu(x)
        return self.fc2(x)


class VisionTransformerBlock(nn.Module):
    """Single ViT block. Keys match timm Block naming exactly."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values: float = 1e-5,
        use_glu_mlp: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        self.norm1 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.attn = ViTAttention(
            dim, num_heads, qkv_bias=qkv_bias,
            dtype=dtype, device=device, operations=operations,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.mlp = ViTMlp(
            dim, mlp_ratio=mlp_ratio, use_glu=use_glu_mlp,
            dtype=dtype, device=device, operations=operations,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, transformer_options={}) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), transformer_options=transformer_options))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Custom ViT matching timm VisionTransformer state_dict keys exactly.

    Replaces TimmViT(timm.models.VisionTransformer) with full operations= support.
    """

    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values: float = 1e-5,
        use_glu_mlp: bool = False,
        num_classes: int = 0,
        intermediate_features_ids: list[int] | None = None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        self.embed_dim = embed_dim
        self.dim_in = in_chans
        self.num_prefix_tokens = 1  # CLS token
        self.intermediate_features_ids = intermediate_features_ids

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, dtype=dtype, device=device, operations=operations,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token and positional embedding (nn.Parameter, not operations)
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim, dtype=dtype, device=device)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim, dtype=dtype, device=device)
        )

        # Pre-norm and patch drop (Identity for DINOv2)
        self.norm_pre = nn.Identity()
        self.patch_drop = nn.Identity()

        # Transformer blocks
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, init_values=init_values, use_glu_mlp=use_glu_mlp,
                dtype=dtype, device=device, operations=operations,
            )
            for _ in range(depth)
        ])

        # Final norm
        self.norm = operations.LayerNorm(embed_dim, dtype=dtype, device=device)

        # Classification head (unused at inference, but needed for checkpoint key compatibility)
        if num_classes > 0:
            self.head = operations.Linear(embed_dim, num_classes, dtype=dtype, device=device)
        else:
            self.head = nn.Identity()

    def reshape_feature(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Discard class token and reshape 1D feature map to a 2D grid."""
        batch_size, seq_len, channel = embeddings.shape
        height, width = self.patch_embed.grid_size

        if self.num_prefix_tokens:
            embeddings = embeddings[:, self.num_prefix_tokens:, :]

        embeddings = embeddings.reshape(batch_size, height, width, channel).permute(0, 3, 1, 2)
        return embeddings

    def forward(
        self, input_tensor: torch.Tensor, transformer_options={}
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Forward pass with intermediate feature extraction."""
        intermediate_features: dict[int, torch.Tensor] = {}

        x = self.patch_embed(input_tensor)

        # Prepend CLS token and add positional embedding
        cls_tokens = self.cls_token.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)

        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for idx, block in enumerate(self.blocks):
            x = block(x, transformer_options=transformer_options)
            if (
                self.intermediate_features_ids is not None
                and idx in self.intermediate_features_ids
            ):
                intermediate_features[idx] = x

        x = self.norm(x)
        x = self.reshape_feature(x)
        return x, intermediate_features

    def internal_resolution(self) -> int:
        """Return the internal image size of the network."""
        if isinstance(self.patch_embed.img_size, tuple):
            return self.patch_embed.img_size[0]
        return self.patch_embed.img_size


# ===========================================================================
# Sliding Pyramid Network
# ===========================================================================


@torch.fx.wrap
def split(
    image: torch.Tensor, overlap_ratio: float = 0.25, patch_size: int = 384
) -> torch.Tensor:
    """Split the input into small patches with sliding window."""
    patch_stride = int(patch_size * (1 - overlap_ratio))
    image_size = image.shape[-1]
    steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1

    x_patch_list = []
    for j in range(steps):
        j0 = j * patch_stride
        j1 = j0 + patch_size
        for i in range(steps):
            i0 = i * patch_stride
            i1 = i0 + patch_size
            x_patch_list.append(image[..., j0:j1, i0:i1])

    return torch.cat(x_patch_list, dim=0)


@torch.fx.wrap
def merge(
    image_patches: torch.Tensor, batch_size: int, padding: int = 3
) -> torch.Tensor:
    """Merge the patched input into a image with sliding window."""
    steps = int(math.sqrt(image_patches.shape[0] // batch_size))
    idx = 0

    output_list = []
    for j in range(steps):
        output_row_list = []
        for i in range(steps):
            output = image_patches[batch_size * idx : batch_size * (idx + 1)]
            if padding != 0:
                if j != 0:
                    output = output[..., padding:, :]
                if i != 0:
                    output = output[..., :, padding:]
                if j != steps - 1:
                    output = output[..., :-padding, :]
                if i != steps - 1:
                    output = output[..., :, :-padding]
            output_row_list.append(output)
            idx += 1
        output_row = torch.cat(output_row_list, dim=-1)
        output_list.append(output_row)
    return torch.cat(output_list, dim=-2)


def _create_spn_project_upsample_block(
    dim_in: int,
    dim_out: int,
    upsample_layers: int,
    dim_intermediate: int | None = None,
    dtype=None,
    device=None,
    operations=None,
) -> nn.Module:
    if operations is None:
        operations = comfy.ops.manual_cast
    if dim_intermediate is None:
        dim_intermediate = dim_out
    blocks: list[nn.Module] = [
        operations.Conv2d(
            in_channels=dim_in, out_channels=dim_intermediate,
            kernel_size=1, stride=1, padding=0, bias=False,
            dtype=dtype, device=device,
        )
    ]
    blocks += [
        operations.ConvTranspose2d(
            in_channels=dim_intermediate if i == 0 else dim_out,
            out_channels=dim_out, kernel_size=2, stride=2, padding=0, bias=False,
            dtype=dtype, device=device,
        )
        for i in range(upsample_layers)
    ]
    return nn.Sequential(*blocks)


class SlidingPyramidNetwork(nn.Module):
    """Sliding Pyramid Network for multi-resolution ViT encodings."""

    def __init__(
        self,
        dims_encoder: list[int],
        patch_encoder: VisionTransformer,
        image_encoder: VisionTransformer,
        use_patch_overlap: bool = True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        self.dim_in = patch_encoder.dim_in
        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder

        base_embed_dim = patch_encoder.embed_dim
        lowres_embed_dim = image_encoder.embed_dim
        self.patch_size = patch_encoder.internal_resolution()
        self.use_patch_overlap = use_patch_overlap

        self.patch_intermediate_features_ids = patch_encoder.intermediate_features_ids
        if (
            not isinstance(self.patch_intermediate_features_ids, list)
            or not len(self.patch_intermediate_features_ids) == 4
        ):
            raise ValueError("Patch intermediate feature ids must be a 4-item list.")

        self.image_intermediate_features_ids = image_encoder.intermediate_features_ids

        self.upsample_latent0 = _create_spn_project_upsample_block(
            dim_in=base_embed_dim, dim_out=self.dims_encoder[0],
            upsample_layers=3, dim_intermediate=self.dims_encoder[1],
            dtype=dtype, device=device, operations=operations,
        )
        self.upsample_latent1 = _create_spn_project_upsample_block(
            dim_in=base_embed_dim, dim_out=self.dims_encoder[1], upsample_layers=2,
            dtype=dtype, device=device, operations=operations,
        )
        self.upsample0 = _create_spn_project_upsample_block(
            dim_in=base_embed_dim, dim_out=self.dims_encoder[2], upsample_layers=1,
            dtype=dtype, device=device, operations=operations,
        )
        self.upsample1 = _create_spn_project_upsample_block(
            dim_in=base_embed_dim, dim_out=self.dims_encoder[3], upsample_layers=1,
            dtype=dtype, device=device, operations=operations,
        )
        self.upsample2 = _create_spn_project_upsample_block(
            dim_in=base_embed_dim, dim_out=self.dims_encoder[4], upsample_layers=1,
            dtype=dtype, device=device, operations=operations,
        )

        self.upsample_lowres = operations.ConvTranspose2d(
            in_channels=lowres_embed_dim, out_channels=self.dims_encoder[4],
            kernel_size=2, stride=2, padding=0, bias=True,
            dtype=dtype, device=device,
        )
        self.fuse_lowres = operations.Conv2d(
            in_channels=(self.dims_encoder[4] + self.dims_encoder[4]),
            out_channels=self.dims_encoder[4],
            kernel_size=1, stride=1, padding=0, bias=True,
            dtype=dtype, device=device,
        )

    def internal_resolution(self) -> int:
        return self.patch_size * 4

    def memory_required(self, input_shape):
        """Estimate peak activation VRAM for SPN encoder + Monodepth decoder.

        The peak occurs during the decoder fusion phase when SPN feature maps
        and decoder intermediates (ResNet Conv2d) are all live.  Memory scales
        with the number of pyramid patches (depends on input spatial dims),
        the ViT embedding width, and dtype — following the same empirical-factor
        pattern as ComfyUI's ``memory_usage_factor`` and VAE estimators.
        """
        import comfy.model_management
        H, W = input_shape[-2], input_shape[-1]
        dtype = next(self.parameters()).dtype
        dtype_size = comfy.model_management.dtype_size(dtype)

        # Patch count from the 3-level pyramid (mirrors forward() logic)
        ps = self.patch_size
        n_x0 = (math.ceil((H - ps) / int(ps * 0.75)) + 1) ** 2 if H > ps else 1
        n_x1 = (math.ceil((H // 2 - ps) / int(ps * 0.5)) + 1) ** 2 if H // 2 > ps else 1
        n_total = n_x0 + n_x1 + 1

        # Per-patch memory covers: chunked ViT forward, encoding storage,
        # merged feature maps, ConvTranspose2d upsample intermediates,
        # Monodepth decoder fusion+ResNet Conv2d, and PyTorch allocator overhead.
        # Calibrated at 1536x1536 bf16: 35 patches, ~6.8 GB peak reserved.
        mem_per_patch = 200 * 1024 * 1024 * (dtype_size / 2) * (self.patch_encoder.embed_dim / 1024)

        return int(n_total * mem_per_patch)

    def _create_pyramid(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = x
        x1 = F.interpolate(x, size=None, scale_factor=0.5, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x, size=None, scale_factor=0.25, mode="bilinear", align_corners=False)
        return x0, x1, x2

    @staticmethod
    def _vram_gb(device):
        return torch.cuda.memory_allocated(device) / (1024 ** 3)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        batch_size = x.shape[0]
        _dev = x.device
        log.warning(f"[SPN] start: {self._vram_gb(_dev):.2f} GB, input={list(x.shape)}")
        x0, x1, x2 = self._create_pyramid(x)

        if self.use_patch_overlap:
            x0_patches = split(x0, overlap_ratio=0.25, patch_size=self.patch_size)
            x1_patches = split(x1, overlap_ratio=0.5, patch_size=self.patch_size)
            x2_patches = x2
            padding = 3
        else:
            x0_patches = split(x0, overlap_ratio=0.0, patch_size=self.patch_size)
            x1_patches = split(x1, overlap_ratio=0.0, patch_size=self.patch_size)
            x2_patches = x2
            padding = 0
        x0_tile_size = x0_patches.shape[0]
        x1_tile_size = x1_patches.shape[0]
        x2_tile_size = x2_patches.shape[0]

        x_pyramid_patches = torch.cat((x0_patches, x1_patches, x2_patches), dim=0)
        del x0, x1, x0_patches, x1_patches  # Free pyramid/patch tensors (x2_patches still needed)
        log.warning(f"[SPN] patches: {self._vram_gb(_dev):.2f} GB, x0={x0_tile_size} x1={x1_tile_size} x2={x2_tile_size} total={x_pyramid_patches.shape[0]}")

        # Dynamic chunking based on free VRAM (same pattern as ComfyUI's attention_split)
        MAX_PATCH_CHUNK = 4
        N = x_pyramid_patches.shape[0]
        mem_free = comfy.model_management.get_free_memory(x_pyramid_patches.device)
        patch_mem_estimate = 200 * 1024 * 1024  # ~200 MB per patch through ViT
        chunk_size = max(1, int(mem_free * 0.8 / patch_mem_estimate))
        chunk_size = min(chunk_size, N, MAX_PATCH_CHUNK)

        # Process first chunk to discover output shapes, then pre-allocate
        first_end = min(chunk_size, N)
        first_enc, first_inter = self.patch_encoder(x_pyramid_patches[:first_end])

        x_pyramid_encodings = torch.empty(
            N, *first_enc.shape[1:], device=first_enc.device, dtype=first_enc.dtype,
        )
        x_pyramid_encodings[:first_end] = first_enc
        del first_enc

        patch_intermediate_features: dict[int, torch.Tensor] = {
            layer_id: torch.empty(
                N, *feat.shape[1:], device=feat.device, dtype=feat.dtype,
            )
            for layer_id, feat in first_inter.items()
        }
        for layer_id, feat in first_inter.items():
            patch_intermediate_features[layer_id][:first_end] = feat
        del first_inter

        log.warning(f"[SPN] after first chunk ({first_end}/{N}): {self._vram_gb(_dev):.2f} GB, chunk_size={chunk_size}")

        for i in range(first_end, N, chunk_size):
            comfy.model_management.throw_exception_if_processing_interrupted()
            end = min(i + chunk_size, N)
            enc, inter = self.patch_encoder(x_pyramid_patches[i:end])
            x_pyramid_encodings[i:end] = enc
            del enc
            for layer_id, feat in inter.items():
                patch_intermediate_features[layer_id][i:end] = feat
            del inter
        del x_pyramid_patches
        # Release cached CUDA blocks from ViT forward passes
        comfy.model_management.soft_empty_cache()
        log.warning(f"[SPN] after all chunks + cache clear: {self._vram_gb(_dev):.2f} GB")

        # Merge intermediate features
        x_latent0_encodings = self.patch_encoder.reshape_feature(
            patch_intermediate_features[self.patch_intermediate_features_ids[0]]
        )
        x_latent0_features = merge(
            x_latent0_encodings[: batch_size * x0_tile_size],
            batch_size=batch_size, padding=padding,
        )

        x_latent1_encodings = self.patch_encoder.reshape_feature(
            patch_intermediate_features[self.patch_intermediate_features_ids[1]]
        )
        x_latent1_features = merge(
            x_latent1_encodings[: batch_size * x0_tile_size],
            batch_size=batch_size, padding=padding,
        )
        del patch_intermediate_features, x_latent0_encodings, x_latent1_encodings
        log.warning(f"[SPN] after merge intermediates: {self._vram_gb(_dev):.2f} GB")

        x0_encodings, x1_encodings, x2_encodings = torch.split(
            x_pyramid_encodings,
            [x0_tile_size, x1_tile_size, x2_tile_size],
            dim=0,
        )
        del x_pyramid_encodings

        x0_features = merge(x0_encodings, batch_size=batch_size, padding=padding)
        del x0_encodings
        x1_features = merge(x1_encodings, batch_size=batch_size, padding=2 * padding)
        del x1_encodings
        x2_features = x2_encodings

        log.warning(f"[SPN] after merge all: {self._vram_gb(_dev):.2f} GB")
        x_lowres_features, _ = self.image_encoder(x2_patches)
        log.warning(f"[SPN] after image_encoder: {self._vram_gb(_dev):.2f} GB")

        x_latent0_features = self.upsample_latent0(x_latent0_features)
        x_latent1_features = self.upsample_latent1(x_latent1_features)
        x0_features = self.upsample0(x0_features)
        x1_features = self.upsample1(x1_features)
        x2_features = self.upsample2(x2_features)
        x_lowres_features = self.upsample_lowres(x_lowres_features)
        x_lowres_features = self.fuse_lowres(
            torch.cat((x2_features, x_lowres_features), dim=1)
        )
        log.warning(f"[SPN] after upsample: {self._vram_gb(_dev):.2f} GB")

        return [
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_lowres_features,
        ]


# ===========================================================================
# Blocks (ResidualBlock, FeatureFusionBlock2d)
# ===========================================================================


class ResidualBlock(nn.Module):
    """Generic residual block."""

    def __init__(self, residual: nn.Module, shortcut: nn.Module | None = None) -> None:
        super().__init__()
        self.residual = residual
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_x = self.residual(x)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return x + delta_x


def residual_block_2d(
    dim_in: int,
    dim_out: int,
    dim_hidden: int | None = None,
    actvn: nn.Module | None = None,
    norm_type: NormLayerName = "noop",
    norm_num_groups: int = 8,
    dilation: int = 1,
    kernel_size: int = 3,
    dtype=None,
    device=None,
    operations=None,
) -> ResidualBlock:
    """Create a simple 2D residual block."""
    if operations is None:
        operations = comfy.ops.manual_cast
    if actvn is None:
        actvn = nn.ReLU()
    if dim_hidden is None:
        dim_hidden = dim_out // 2

    padding = (dilation * (kernel_size - 1)) // 2

    def _create_block(d_in: int, d_out: int) -> list[nn.Module]:
        layers: list[nn.Module] = [
            norm_layer_2d(d_in, norm_type, num_groups=norm_num_groups,
                          dtype=dtype, device=device, operations=operations),
            actvn,
            operations.Conv2d(
                d_in, d_out, kernel_size=kernel_size, stride=1,
                dilation=dilation, padding=padding,
                dtype=dtype, device=device,
            ),
        ]
        return layers

    residual = nn.Sequential(
        *_create_block(dim_in, dim_hidden),
        *_create_block(dim_hidden, dim_out),
    )
    shortcut = None
    if dim_in != dim_out:
        shortcut = operations.Conv2d(dim_in, dim_out, 1, dtype=dtype, device=device)

    return ResidualBlock(residual, shortcut)


class FeatureFusionBlock2d(nn.Module):
    """Feature fusion for DPT."""

    deconv: nn.Module

    def __init__(
        self,
        dim_in: int,
        dim_out: int | None = None,
        upsampling_mode: UpsamplingMode | None = None,
        batch_norm: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        if dim_out is None:
            dim_out = dim_in
        self.resnet1 = self._residual_block(dim_in, batch_norm,
                                            dtype=dtype, device=device, operations=operations)
        self.resnet2 = self._residual_block(dim_in, batch_norm,
                                            dtype=dtype, device=device, operations=operations)

        if upsampling_mode is not None:
            self.deconv = upsampling_layer(upsampling_mode, scale_factor=2, dim_in=dim_in,
                                           dtype=dtype, device=device, operations=operations)
        else:
            self.deconv = nn.Sequential()

        self.out_conv = operations.Conv2d(
            dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=True,
            dtype=dtype, device=device,
        )

    def forward(self, x0: torch.Tensor, x1: torch.Tensor | None = None) -> torch.Tensor:
        x = x0
        if x1 is not None:
            res = self.resnet1(x1)
            x = x + res
        x = self.resnet2(x)
        x = self.deconv(x)
        x = self.out_conv(x)
        return x

    @staticmethod
    def _residual_block(
        num_features: int,
        batch_norm: bool,
        dtype=None,
        device=None,
        operations=None,
    ) -> ResidualBlock:
        if operations is None:
            operations = comfy.ops.manual_cast
        def _create_block(dim: int, bn: bool) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.ReLU(False),
                operations.Conv2d(
                    num_features, num_features, kernel_size=3, stride=1, padding=1,
                    bias=not bn, dtype=dtype, device=device,
                ),
            ]
            if bn:
                layers.append(nn.BatchNorm2d(dim))
            return layers

        residual = nn.Sequential(
            *_create_block(dim=num_features, bn=batch_norm),
            *_create_block(dim=num_features, bn=batch_norm),
        )
        return ResidualBlock(residual)


# ===========================================================================
# MultiresConvDecoder
# ===========================================================================


class MultiresConvDecoder(nn.Module):
    """Decoder for multi-resolution encodings."""

    def __init__(
        self,
        dims_encoder: list[int],
        dims_decoder: list[int] | int,
        upsampling_mode: UpsamplingMode = "transposed_conv",
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        self.dims_encoder = list(dims_encoder)

        if isinstance(dims_decoder, int):
            self.dims_decoder = [dims_decoder] * len(self.dims_encoder)
        else:
            self.dims_decoder = list(dims_decoder)

        if len(self.dims_decoder) != len(self.dims_encoder):
            raise ValueError("Received dims_encoder and dims_decoder of different sizes.")

        self.dim_out = self.dims_decoder[0]
        num_encoders = len(self.dims_encoder)

        conv0: nn.Module = (
            operations.Conv2d(self.dims_encoder[0], self.dims_decoder[0],
                              kernel_size=1, bias=False, dtype=dtype, device=device)
            if self.dims_encoder[0] != self.dims_decoder[0]
            else nn.Identity()
        )

        convs: list[nn.Module] = [conv0]
        for i in range(1, num_encoders):
            convs.append(
                operations.Conv2d(
                    self.dims_encoder[i], self.dims_decoder[i],
                    kernel_size=3, stride=1, padding=1, bias=False,
                    dtype=dtype, device=device,
                )
            )
        self.convs = nn.ModuleList(convs)

        fusions = []
        for i in range(num_encoders):
            fusions.append(
                FeatureFusionBlock2d(
                    dim_in=self.dims_decoder[i],
                    dim_out=self.dims_decoder[i - 1] if i != 0 else self.dim_out,
                    upsampling_mode=upsampling_mode if i != 0 else None,
                    batch_norm=False,
                    dtype=dtype, device=device, operations=operations,
                )
            )
        self.fusions = nn.ModuleList(fusions)

    def forward(self, encodings: list[torch.Tensor]) -> torch.Tensor:
        num_levels = len(encodings)
        num_encoders = len(self.dims_encoder)
        if num_levels != num_encoders:
            raise ValueError(
                f"Encoder output levels={num_levels} mismatch with expected={num_encoders}."
            )

        features = self.convs[-1](encodings[-1])
        features = self.fusions[-1](features)
        for i in range(num_levels - 2, -1, -1):
            features_i = self.convs[i](encodings[i])
            features = self.fusions[i](features, features_i)
        return features


# ===========================================================================
# Normalizers
# ===========================================================================


class AffineRangeNormalizer(nn.Module):
    """Perform linear mapping from input_range to output_range."""

    def __init__(
        self,
        input_range: tuple[float, float],
        output_range: tuple[float, float] = (0, 1),
    ):
        super().__init__()
        input_min, input_max = input_range
        output_min, output_max = output_range
        if input_max <= input_min:
            raise ValueError(f"Invalid input_range: {input_range}")
        if output_max <= output_min:
            raise ValueError(f"Invalid output_range: {output_range}")
        self.scale = (output_max - output_min) / (input_max - input_min)
        self.bias = output_min - input_min * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale != 1.0:
            x = x * self.scale
        if self.bias != 0.0:
            x = x + self.bias
        return x


class MeanStdNormalizer(nn.Module):
    """Normalizing image input by mean and std."""

    mean: torch.Tensor
    std_inv: torch.Tensor

    def __init__(
        self,
        mean: Union[Sequence[float], torch.Tensor],
        std: Union[Sequence[float], torch.Tensor],
    ):
        super().__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.as_tensor(mean).view(-1, 1, 1)
        if not isinstance(std, torch.Tensor):
            std = torch.as_tensor(std).view(-1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std_inv", 1.0 / std)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) * self.std_inv


# ===========================================================================
# Monodepth
# ===========================================================================


class MonodepthDensePredictionTransformer(nn.Module):
    """Dense Prediction Transformer for monodepth."""

    def __init__(
        self,
        encoder: SlidingPyramidNetwork,
        decoder: MultiresConvDecoder,
        last_dims: tuple[int, int],
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        self.normalizer = AffineRangeNormalizer(
            input_range=(0, 1), output_range=(-1, 1)
        )
        self.encoder = encoder
        self.decoder = decoder

        dim_decoder = decoder.dim_out
        self.head = nn.Sequential(
            operations.Conv2d(dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1,
                              dtype=dtype, device=device),
            operations.ConvTranspose2d(
                in_channels=dim_decoder // 2, out_channels=dim_decoder // 2,
                kernel_size=2, stride=2, padding=0, bias=True,
                dtype=dtype, device=device,
            ),
            operations.Conv2d(
                dim_decoder // 2, last_dims[0], kernel_size=3, stride=1, padding=1,
                dtype=dtype, device=device,
            ),
            nn.ReLU(True),
            operations.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0,
                              dtype=dtype, device=device),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encodings = self.encoder(self.normalizer(image))
        num_encoder_features = len(self.encoder.dims_encoder)
        features = self.decoder(encodings[:num_encoder_features])
        disparity = self.head(features)
        return disparity

    def internal_resolution(self) -> int:
        return self.encoder.internal_resolution()


class MonodepthWithEncodingAdaptor(nn.Module):
    """Monodepth model with feature maps."""

    def __init__(
        self,
        monodepth_predictor: MonodepthDensePredictionTransformer,
        return_encoder_features: bool,
        return_decoder_features: bool,
        num_monodepth_layers: int,
        sorting_monodepth: bool,
    ):
        super().__init__()
        self.monodepth_predictor = monodepth_predictor
        self.return_encoder_features = return_encoder_features
        self.return_decoder_features = return_decoder_features
        self.num_monodepth_layers = num_monodepth_layers
        self.sorting_monodepth = sorting_monodepth

    @staticmethod
    def _vram_gb(device):
        return torch.cuda.memory_allocated(device) / (1024 ** 3)

    def forward(self, image: torch.Tensor) -> MonodepthOutput:
        _dev = image.device
        inputs = self.monodepth_predictor.normalizer(image)
        encoder_output = self.monodepth_predictor.encoder(inputs)
        # Release cached CUDA blocks from SPN's heavy ViT processing
        comfy.model_management.soft_empty_cache()
        log.warning(f"[Monodepth] after SPN encoder + cache clear: {self._vram_gb(_dev):.2f} GB")

        num_encoder_features = len(self.monodepth_predictor.encoder.dims_encoder)
        encoder_features = encoder_output[:num_encoder_features]
        intermediate_features = encoder_output[num_encoder_features:]
        del encoder_output
        decoder_features = self.monodepth_predictor.decoder(encoder_features)
        log.warning(f"[Monodepth] after decoder: {self._vram_gb(_dev):.2f} GB")
        disparity = self.monodepth_predictor.head(decoder_features)

        if self.num_monodepth_layers == 2 and self.sorting_monodepth:
            first_layer_disparity = disparity.max(dim=1, keepdims=True).values
            second_layer_disparity = disparity.min(dim=1, keepdims=True).values
            disparity = torch.cat([first_layer_disparity, second_layer_disparity], dim=1)

        output_features: list[torch.Tensor] = []
        if self.return_encoder_features:
            output_features.extend(encoder_features)
        if self.return_decoder_features:
            output_features.append(decoder_features)

        return MonodepthOutput(
            disparity=disparity,
            encoder_features=encoder_features,
            decoder_features=decoder_features,
            output_features=output_features,
            intermediate_features=intermediate_features,
        )

    def get_feature_dims(self) -> list[int]:
        dims: list[int] = []
        if self.return_encoder_features:
            dims.extend(self.monodepth_predictor.encoder.dims_encoder)
        if self.return_decoder_features:
            dims.append(self.monodepth_predictor.decoder.dim_out)
        return dims

    def internal_resolution(self) -> int:
        return self.monodepth_predictor.internal_resolution()

    def replicate_head(self, num_repeat: int):
        """Replicate the last convolution layer for multi-layer depth."""
        conv_last = copy.deepcopy(self.monodepth_predictor.head[4])
        self.monodepth_predictor.head[4].out_channels = num_repeat
        self.monodepth_predictor.head[4].weight = nn.Parameter(
            conv_last.weight.repeat(num_repeat, 1, 1, 1)
        )
        self.monodepth_predictor.head[4].bias = nn.Parameter(
            conv_last.bias.repeat(num_repeat)
        )


# ===========================================================================
# Gaussian Decoder
# ===========================================================================


def _create_project_upsample_block(
    dim_in: int,
    dim_out: int,
    upsample_layers: int,
    dim_intermediate: int | None = None,
    dtype=None,
    device=None,
    operations=None,
) -> nn.Module:
    if operations is None:
        operations = comfy.ops.manual_cast
    if dim_intermediate is None:
        dim_intermediate = dim_out
    blocks: list[nn.Module] = [
        operations.Conv2d(
            in_channels=dim_in, out_channels=dim_intermediate,
            kernel_size=1, stride=1, padding=0, bias=False,
            dtype=dtype, device=device,
        )
    ]
    blocks += [
        operations.ConvTranspose2d(
            in_channels=dim_intermediate if i == 0 else dim_out,
            out_channels=dim_out, kernel_size=2, stride=2, padding=0, bias=False,
            dtype=dtype, device=device,
        )
        for i in range(upsample_layers)
    ]
    return nn.Sequential(*blocks)


class SkipConvBackbone(nn.Module):
    """A wrapper around a conv layer that behaves like a BaseBackbone."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int,
        stride_out: int,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        self.stride_out = stride_out
        if stride_out == 1 and kernel_size != 1:
            raise ValueError("We only support kernel_size = 1 if stride_out is 1.")
        padding = (kernel_size - 1) // 2
        self.conv = operations.Conv2d(
            dim_in, dim_out, kernel_size=kernel_size, stride=stride_out, padding=padding,
            dtype=dtype, device=device,
        )

    def forward(
        self, input_features: torch.Tensor, encodings: list[torch.Tensor] | None = None
    ) -> ImageFeatures:
        output = self.conv(input_features)
        return ImageFeatures(texture_features=output, geometry_features=output)

    @property
    def stride(self) -> int:
        return self.stride_out


class GaussianDensePredictionTransformer(nn.Module):
    """Dense Prediction Transformer for Gaussian features."""

    norm_type: NormLayerName

    def __init__(
        self,
        decoder: MultiresConvDecoder,
        dim_in: int,
        dim_out: int,
        stride_out: int,
        image_encoder_params,  # GaussianDecoderParams
        image_encoder_type: DPTImageEncoderType = "skip_conv",
        norm_type: NormLayerName = "group_norm",
        norm_num_groups: int = 8,
        use_depth_input: bool = True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        self.decoder = decoder
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.stride_out = stride_out
        self.norm_type = norm_type
        self.norm_num_groups = norm_num_groups
        self.use_depth_input = use_depth_input
        self.image_encoder_type = image_encoder_type

        enc_dim_in = self.dim_in if use_depth_input else self.dim_in - 1
        image_encoder_params.dim_in = enc_dim_in
        image_encoder_params.dim_out = decoder.dim_out
        self.image_encoder = self._create_image_encoder(
            image_encoder_params, stride_out,
            dtype=dtype, device=device, operations=operations,
        )

        self.fusion = FeatureFusionBlock2d(decoder.dim_out,
                                           dtype=dtype, device=device, operations=operations)

        if stride_out == 1:
            self.upsample = _create_project_upsample_block(
                decoder.dim_out, decoder.dim_out, upsample_layers=1,
                dtype=dtype, device=device, operations=operations,
            )
        elif stride_out == 2:
            self.upsample = nn.Identity()
        else:
            raise ValueError("We only support stride is 1 or 2 for DPT backbone.")

        self.texture_head = self._create_head(
            dim_decoder=decoder.dim_out, dim_out=self.dim_out,
            dtype=dtype, device=device, operations=operations,
        )
        self.geometry_head = self._create_head(
            dim_decoder=decoder.dim_out, dim_out=self.dim_out,
            dtype=dtype, device=device, operations=operations,
        )

    def _create_head(
        self, dim_decoder: int, dim_out: int,
        dtype=None, device=None, operations=None,
    ) -> nn.Module:
        if operations is None:
            operations = comfy.ops.manual_cast
        return nn.Sequential(
            residual_block_2d(
                dim_in=dim_decoder, dim_out=dim_decoder, dim_hidden=dim_decoder // 2,
                norm_type=self.norm_type, norm_num_groups=self.norm_num_groups,
                dtype=dtype, device=device, operations=operations,
            ),
            residual_block_2d(
                dim_in=dim_decoder, dim_hidden=dim_decoder // 2, dim_out=dim_decoder,
                norm_type=self.norm_type, norm_num_groups=self.norm_num_groups,
                dtype=dtype, device=device, operations=operations,
            ),
            nn.ReLU(),
            operations.Conv2d(dim_decoder, dim_out, kernel_size=1, stride=1,
                              dtype=dtype, device=device),
            nn.ReLU(),
        )

    def _create_image_encoder(
        self, image_encoder_params, stride_out: int,
        dtype=None, device=None, operations=None,
    ) -> nn.Module:
        if operations is None:
            operations = comfy.ops.manual_cast
        if self.image_encoder_type == "skip_conv":
            return SkipConvBackbone(
                image_encoder_params.dim_in, image_encoder_params.dim_out,
                kernel_size=3 if stride_out != 1 else 1, stride_out=stride_out,
                dtype=dtype, device=device, operations=operations,
            )
        elif self.image_encoder_type == "skip_conv_kernel2":
            return SkipConvBackbone(
                image_encoder_params.dim_in, image_encoder_params.dim_out,
                kernel_size=stride_out, stride_out=stride_out,
                dtype=dtype, device=device, operations=operations,
            )
        else:
            raise ValueError(f"Unsupported image encoder type: {self.image_encoder_type}")

    def forward(
        self, input_features: torch.Tensor, encodings: list[torch.Tensor]
    ) -> ImageFeatures:
        features = self.decoder(encodings).contiguous()
        features = self.upsample(features)

        if self.use_depth_input:
            skip_features = self.image_encoder(input_features).texture_features
        else:
            skip_features = self.image_encoder(input_features[:, :3].contiguous())
        features = self.fusion(features, skip_features)

        texture_features = self.texture_head(features)
        geometry_features = self.geometry_head(features)

        return ImageFeatures(
            texture_features=texture_features,
            geometry_features=geometry_features,
        )

    @property
    def stride(self) -> int:
        return self.stride_out


# ===========================================================================
# Prediction Head
# ===========================================================================


class DirectPredictionHead(nn.Module):
    """Decodes features into delta values using convolutions."""

    def __init__(
        self,
        feature_dim: int,
        num_layers: int,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        self.num_layers = num_layers
        self.geometry_prediction_head = operations.Conv2d(
            feature_dim, 3 * num_layers, 1, dtype=dtype, device=device,
        )
        self.texture_prediction_head = operations.Conv2d(
            feature_dim, (14 - 3) * num_layers, 1, dtype=dtype, device=device,
        )

    def forward(self, image_features: ImageFeatures) -> torch.Tensor:
        delta_values_geometry = self.geometry_prediction_head(image_features.geometry_features)
        delta_values_texture = self.texture_prediction_head(image_features.texture_features)
        delta_values_geometry = delta_values_geometry.unflatten(1, (3, self.num_layers))
        delta_values_texture = delta_values_texture.unflatten(1, (14 - 3, self.num_layers))
        return torch.cat([delta_values_geometry, delta_values_texture], dim=1)


# ===========================================================================
# Initializer
# ===========================================================================


def _create_base_xy(
    depth: torch.Tensor, stride: int, num_layers: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create base x and y coordinates for Gaussians in NDC space."""
    device = depth.device
    batch_size, _, image_height, image_width = depth.shape
    xx = torch.arange(0.5 * stride, image_width, stride, device=device)
    yy = torch.arange(0.5 * stride, image_height, stride, device=device)
    xx = 2 * xx / image_width - 1.0
    yy = 2 * yy / image_height - 1.0

    xx, yy = torch.meshgrid(xx, yy, indexing="xy")
    base_x_ndc = xx[None, None, None].repeat(batch_size, 1, num_layers, 1, 1)
    base_y_ndc = yy[None, None, None].repeat(batch_size, 1, num_layers, 1, 1)
    return base_x_ndc, base_y_ndc


def _create_base_scale(disparity: torch.Tensor, disparity_scale_factor: float) -> torch.Tensor:
    """Create base scale for Gaussians."""
    inverse_disparity = torch.ones_like(disparity) / disparity
    return inverse_disparity * disparity_scale_factor


def _rescale_depth(
    depth: torch.Tensor, depth_min: float = 1.0, depth_max: float = 1e2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rescale a depth image tensor."""
    current_depth_min = depth.flatten(depth.ndim - 3).min(dim=-1).values
    depth_factor = depth_min / (current_depth_min + 1e-6)
    depth = (depth * depth_factor[..., None, None, None]).clamp(max=depth_max)
    return depth, depth_factor


class MultiLayerInitializer(nn.Module):
    """Initialize Gaussians with multilayer representation."""

    def __init__(
        self,
        num_layers: int,
        stride: int,
        base_depth: float,
        scale_factor: float,
        disparity_factor: float,
        color_option: ColorInitOption = "first_layer",
        first_layer_depth_option: DepthInitOption = "surface_min",
        rest_layer_depth_option: DepthInitOption = "surface_min",
        normalize_depth: bool = True,
        feature_input_stop_grad: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.stride = stride
        self.base_depth = base_depth
        self.scale_factor = scale_factor
        self.disparity_factor = disparity_factor
        self.color_option = color_option
        self.first_layer_depth_option = first_layer_depth_option
        self.rest_layer_depth_option = rest_layer_depth_option
        self.normalize_depth = normalize_depth
        self.feature_input_stop_grad = feature_input_stop_grad

    def prepare_feature_input(self, image: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if self.feature_input_stop_grad:
            image = image.detach()
            depth = depth.detach()
        normalized_disparity = self.disparity_factor / depth
        features_in = torch.cat([image, normalized_disparity], dim=1)
        features_in = 2.0 * features_in - 1.0
        return features_in

    def forward(self, image: torch.Tensor, depth: torch.Tensor) -> InitializerOutput:
        image = image.contiguous()
        depth = depth.contiguous()
        device = depth.device
        batch_size, _, image_height, image_width = depth.shape
        base_height, base_width = image_height // self.stride, image_width // self.stride

        global_scale: torch.Tensor | None = None
        if self.normalize_depth:
            depth, depth_factor = _rescale_depth(depth)
            global_scale = 1.0 / depth_factor

        def _create_disparity_layers(n_layers: int = 1) -> torch.Tensor:
            disparity = torch.linspace(1.0 / self.base_depth, 0.0, n_layers + 1, device=device)
            return disparity[None, None, :-1, None, None].repeat(
                batch_size, 1, 1, base_height, base_width
            )

        def _create_surface_layer(d: torch.Tensor, depth_pooling_mode: str) -> torch.Tensor:
            disp = 1.0 / d
            if depth_pooling_mode == "min":
                disp = torch.max_pool2d(disp, self.stride, self.stride)
            elif depth_pooling_mode == "max":
                disp = -torch.max_pool2d(-disp, self.stride, self.stride)
            else:
                raise ValueError(f"Invalid depth pooling mode {depth_pooling_mode}.")
            return disp[:, :, None, :, :]

        if self.first_layer_depth_option == "surface_min":
            first_disparity = _create_surface_layer(depth[:, 0:1], "min")
        elif self.first_layer_depth_option == "surface_max":
            first_disparity = _create_surface_layer(depth[:, 0:1], "max")
        elif self.first_layer_depth_option in ("base_depth", "linear_disparity"):
            first_disparity = _create_disparity_layers()
        else:
            raise ValueError(f"Unknown depth init option: {self.first_layer_depth_option}.")

        if self.num_layers == 1:
            disparity = first_disparity
        else:
            following_depth = depth if depth.shape[1] == 1 else depth[:, 1:]
            if self.rest_layer_depth_option == "surface_min":
                following_disparity = _create_surface_layer(following_depth, "min")
            elif self.rest_layer_depth_option == "surface_max":
                following_disparity = _create_surface_layer(following_depth, "max")
            elif self.rest_layer_depth_option == "base_depth":
                following_disparity = torch.cat(
                    [_create_disparity_layers() for _ in range(self.num_layers - 1)], dim=2,
                )
            elif self.rest_layer_depth_option == "linear_disparity":
                following_disparity = _create_disparity_layers(self.num_layers - 1)
            else:
                raise ValueError(f"Unknown depth init option: {self.rest_layer_depth_option}.")
            disparity = torch.cat([first_disparity, following_disparity], dim=2)

        base_x_ndc, base_y_ndc = _create_base_xy(depth, self.stride, self.num_layers)
        disparity_scale_factor = 2 * self.scale_factor * self.stride / float(image_width)
        base_scales = _create_base_scale(disparity, disparity_scale_factor)

        base_quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        base_quaternions = base_quaternions[None, :, None, None, None]
        base_opacities = torch.tensor([min(1.0 / self.num_layers, 0.5)], device=device)
        base_colors = torch.empty(
            batch_size, 3, self.num_layers, base_height, base_width, device=device
        ).fill_(0.5)

        if self.color_option == "none":
            pass
        elif self.color_option == "first_layer":
            base_colors[:, :, 0] = F.avg_pool2d(image, self.stride, self.stride)
        elif self.color_option == "all_layers":
            temp = F.avg_pool2d(image, self.stride, self.stride)
            base_colors = temp[:, :, None, :, :].repeat(1, 1, self.num_layers, 1, 1)
        else:
            raise ValueError(f"Unknown color init option: {self.color_option}.")

        features_in = self.prepare_feature_input(image, depth)
        base_gaussians = GaussianBaseValues(
            mean_x_ndc=base_x_ndc, mean_y_ndc=base_y_ndc,
            mean_inverse_z_ndc=disparity, scales=base_scales,
            quaternions=base_quaternions, colors=base_colors, opacities=base_opacities,
        )
        return InitializerOutput(
            gaussian_base_values=base_gaussians,
            feature_input=features_in,
            global_scale=global_scale,
        )


# ===========================================================================
# Gaussian Composer
# ===========================================================================


def _get_scale_activation_constant(max_scale: float, min_scale: float) -> tuple[float, float]:
    constant_a = (max_scale - min_scale) / (1 - min_scale) / (max_scale - 1)
    x = torch.tensor((1.0 - min_scale) / (max_scale - min_scale))
    constant_b = torch.log(x / (1 - x)).item()
    return constant_a, constant_b


class GaussianComposer(nn.Module):
    """Converts base values and deltas into Gaussians. Pure tensor ops, no learnable layers."""

    color_activation_type: str
    opacity_activation_type: str

    def __init__(
        self,
        delta_factor,  # DeltaFactor
        min_scale: float,
        max_scale: float,
        color_activation_type: str,
        opacity_activation_type: str,
        color_space: ColorSpace,
        base_scale_on_predicted_mean: bool,
        scale_factor: int = 1,
    ) -> None:
        super().__init__()
        self.delta_factor = delta_factor
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.color_activation_type = color_activation_type
        self.opacity_activation_type = opacity_activation_type
        self.color_space = color_space
        self.scale_factor = scale_factor
        self.base_scale_on_predicted_mean = base_scale_on_predicted_mean

    def upsample_delta_value(self, delta: torch.Tensor, scale_factor: int = 1):
        B, C, N, H, W = delta.shape
        new_height = H * scale_factor
        new_width = W * scale_factor
        return F.interpolate(
            delta.view(B, C * N, H, W), scale_factor=scale_factor,
        ).view(B, C, N, new_height, new_width)

    def forward(
        self,
        delta: torch.Tensor,
        base_values: GaussianBaseValues,
        global_scale: torch.Tensor | None = None,
        flatten_output: bool = True,
    ) -> Gaussians3D:
        scale_factor = self.scale_factor
        actual_scale_factor = base_values.mean_x_ndc.shape[-1] // delta.shape[-1]
        if scale_factor != 1 and actual_scale_factor != 1:
            delta = self.upsample_delta_value(delta, scale_factor)

        mean_vectors = self._forward_mean(base_values, delta)
        base_scales = (
            (base_values.scales * base_values.mean_inverse_z_ndc * mean_vectors[:, 2:3, ...])
            if self.base_scale_on_predicted_mean
            else base_values.scales
        )
        singular_values = self._scale_activation(
            base_scales, delta[:, 3:6], self.min_scale, self.max_scale,
        )
        quaternions = self._quaternion_activation(base_values.quaternions, delta[:, 6:10])
        colors = self._color_activation(base_values.colors, delta[:, 10:13])
        opacities = self._opacity_activation(base_values.opacities, delta[:, 13])

        if flatten_output:
            mean_vectors = mean_vectors.permute(0, 2, 3, 4, 1).flatten(1, 3)
            singular_values = singular_values.permute(0, 2, 3, 4, 1).flatten(1, 3)
            quaternions = quaternions.permute(0, 2, 3, 4, 1).flatten(1, 3)
            colors = colors.permute(0, 2, 3, 4, 1).flatten(1, 3)
            opacities = opacities.flatten(1, 3)

        if global_scale is not None:
            mean_vectors = global_scale[:, None, None] * mean_vectors
            singular_values = global_scale[:, None, None] * singular_values

        return Gaussians3D(
            mean_vectors=mean_vectors, singular_values=singular_values,
            quaternions=quaternions, colors=colors, opacities=opacities,
        )

    def _forward_mean(self, base_values: GaussianBaseValues, delta: torch.Tensor) -> torch.Tensor:
        delta_factor = torch.tensor(
            [self.delta_factor.xy, self.delta_factor.xy, self.delta_factor.z],
            device=delta.device,
        )[None, :, None, None, None]

        dtype = base_values.mean_x_ndc.dtype
        device = base_values.mean_x_ndc.device
        target_shape = (1, 3, 1, 1, 1)
        mean_x_mask = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).reshape(target_shape)
        mean_y_mask = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).reshape(target_shape)
        mean_z_mask = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).reshape(target_shape)

        mean_vectors_ndc = (
            base_values.mean_x_ndc.repeat(target_shape) * mean_x_mask
            + base_values.mean_y_ndc.repeat(target_shape) * mean_y_mask
            + base_values.mean_inverse_z_ndc.repeat(target_shape) * mean_z_mask
        )
        return self._mean_activation(mean_vectors_ndc, delta_factor * delta[:, :3])

    def _mean_activation(self, base: torch.Tensor, learned_delta: torch.Tensor) -> torch.Tensor:
        xx = base[:, 0:1] + learned_delta[:, 0:1]
        yy = base[:, 1:2] + learned_delta[:, 1:2]
        a = base[:, 2:3]
        b = learned_delta[:, 2:3]
        inverse_zz = F.softplus(_inverse_softplus(a) + b)
        zz = 1.0 / (inverse_zz + 1e-3)
        return torch.cat([zz * xx, zz * yy, zz], dim=1)

    def _scale_activation(
        self, base: torch.Tensor, learned_delta: torch.Tensor,
        min_scale: float, max_scale: float,
    ) -> torch.Tensor:
        constant_a, constant_b = _get_scale_activation_constant(max_scale, min_scale)
        scale_factor = (max_scale - min_scale) * torch.sigmoid(
            constant_a * self.delta_factor.scale * learned_delta + constant_b
        ) + min_scale
        return base * scale_factor

    def _quaternion_activation(
        self, base: torch.Tensor, learned_delta: torch.Tensor
    ) -> torch.Tensor:
        return base + self.delta_factor.quaternion * learned_delta

    def _color_activation(self, base: torch.Tensor, learned_delta: torch.Tensor) -> torch.Tensor:
        if self.color_activation_type == "sigmoid":
            base = torch.clamp(base, min=0.01, max=0.99)
        elif self.color_activation_type in ("exp", "softplus"):
            base = torch.clamp(base, min=0.01)
        fwd, inv = _ACTIVATIONS[self.color_activation_type]
        colors = fwd(inv(base) + self.delta_factor.color * learned_delta)
        if self.color_space == "linearRGB":
            colors = sRGB2linearRGB(colors)
        return colors

    def _opacity_activation(self, base: torch.Tensor, learned_delta: torch.Tensor) -> torch.Tensor:
        fwd, inv = _ACTIVATIONS[self.opacity_activation_type]
        return fwd(inv(base) + self.delta_factor.opacity * learned_delta)


# ===========================================================================
# UNet Encoder / Decoder (used by LearnedAlignment)
# ===========================================================================


class UNetEncoder(nn.Module):
    """Encoder of UNet model."""

    def __init__(
        self,
        dim_in: int,
        width: list[int] | int,
        steps: int = 6,
        norm_type: NormLayerName = "group_norm",
        norm_num_groups: int = 8,
        blocks_per_layer: int = 2,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        if blocks_per_layer < 1:
            raise ValueError("blocks_per_layer must be >= 1.")

        self.dim_in = dim_in
        self.width = width
        self.num_steps = steps

        self.convs_down = nn.ModuleList()

        if isinstance(width, int):
            self.output_dims = [width << i for i in range(0, steps + 1)]
        else:
            if len(width) != (steps + 1):
                raise ValueError("Length of width should match the steps for UNetEncoder.")
            self.output_dims = list(width)

        self.conv_in = nn.Sequential(
            operations.Conv2d(self.dim_in, self.output_dims[0], 3, stride=1, padding=1,
                              dtype=dtype, device=device),
            norm_layer_2d(self.output_dims[0], norm_type, num_groups=norm_num_groups,
                          dtype=dtype, device=device, operations=operations),
            nn.ReLU(),
        )

        for i_step in range(steps):
            input_width = self.output_dims[i_step]
            current_width = self.output_dims[i_step + 1]
            convs_down_i = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                residual_block_2d(
                    input_width, current_width,
                    norm_type=norm_type, norm_num_groups=norm_num_groups,
                    dtype=dtype, device=device, operations=operations,
                ),
                *[
                    residual_block_2d(
                        current_width, current_width,
                        norm_type=norm_type, norm_num_groups=norm_num_groups,
                        dtype=dtype, device=device, operations=operations,
                    )
                    for _ in range(blocks_per_layer - 1)
                ],
            )
            self.convs_down.append(convs_down_i)

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        features = []
        feat_i = self.conv_in(input)
        features.append(feat_i)
        for conv_down in self.convs_down:
            feat_i = conv_down(feat_i)
            features.append(feat_i)
        return features

    @property
    def out_width(self) -> int:
        return self.output_dims[-1]


class UNetDecoder(nn.Module):
    """Decoder of UNet model."""

    def __init__(
        self,
        dim_out: int,
        width: list[int] | int,
        steps: int = 5,
        norm_type: NormLayerName = "group_norm",
        norm_num_groups: int = 8,
        blocks_per_layer: int = 2,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        if blocks_per_layer < 1:
            raise ValueError("blocks_per_layer must be >= 1.")

        self.dim_out = dim_out
        self.convs_up = nn.ModuleList()

        if isinstance(width, int):
            self.input_dims = [width >> i for i in range(0, steps + 1)]
        else:
            self.input_dims = list(width)[::-1][: steps + 1]

        for i_step in range(steps):
            input_width = self.input_dims[i_step]
            current_width = self.input_dims[i_step + 1]
            convs_up_i = nn.Sequential(
                nn.Upsample(scale_factor=2),
                residual_block_2d(
                    input_width * (1 if i_step == 0 else 2),
                    current_width,
                    norm_type=norm_type, norm_num_groups=norm_num_groups,
                    dtype=dtype, device=device, operations=operations,
                ),
                *[
                    residual_block_2d(
                        current_width, current_width,
                        norm_type=norm_type, norm_num_groups=norm_num_groups,
                        dtype=dtype, device=device, operations=operations,
                    )
                    for _ in range(blocks_per_layer - 1)
                ],
            )
            self.convs_up.append(convs_up_i)

        last_width = self.input_dims[-1]
        self.conv_out = nn.Sequential(
            norm_layer_2d(last_width * 2, norm_type, num_groups=norm_num_groups,
                          dtype=dtype, device=device, operations=operations),
            nn.ReLU(),
            operations.Conv2d(last_width * 2, dim_out, 1, dtype=dtype, device=device),
            norm_layer_2d(dim_out, norm_type, num_groups=norm_num_groups,
                          dtype=dtype, device=device, operations=operations),
            nn.ReLU(),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        i_feature_layer = len(features) - 1
        out = self.convs_up[0](features[i_feature_layer])
        i_feature_layer -= 1
        for conv_up in self.convs_up[1:]:
            out = conv_up(torch.cat([out, features[i_feature_layer]], dim=1))
            i_feature_layer -= 1
        out = self.conv_out(torch.cat([out, features[i_feature_layer]], dim=1))
        return out


# ===========================================================================
# Learned Alignment
# ===========================================================================


class LearnedAlignment(nn.Module):
    """Aligns tensors using a UNet."""

    def __init__(
        self,
        steps: int = 4,
        stride: int = 8,
        base_width: int = 16,
        depth_decoder_features: bool = False,
        depth_decoder_dim: int = 256,
        activation_type: str = "exp",
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        if operations is None:
            operations = comfy.ops.manual_cast
        self._act_fwd, self._act_inv = _ACTIVATIONS[activation_type]
        bias_value = self._act_inv(torch.tensor(1.0))

        self.depth_decoder_features = depth_decoder_features
        dim_in = 2 + depth_decoder_dim if depth_decoder_features else 2

        def is_power_of_two(n: int) -> bool:
            if n <= 0:
                return False
            return (n & (n - 1)) == 0

        if not is_power_of_two(stride):
            raise ValueError(f"Stride {stride} is not a power of two.")

        steps_decoder = steps - int(math.log2(stride))
        if steps_decoder < 1:
            raise ValueError(f"{steps_decoder} must be >= 1.")
        widths = [min(base_width << i, 1024) for i in range(steps + 1)]
        self.encoder = UNetEncoder(
            dim_in=dim_in, width=widths, steps=steps, norm_num_groups=4,
            dtype=dtype, device=device, operations=operations,
        )
        self.decoder = UNetDecoder(
            dim_out=widths[0], width=widths, steps=steps_decoder, norm_num_groups=4,
            dtype=dtype, device=device, operations=operations,
        )
        self.conv_out = operations.Conv2d(widths[0], 1, 1, bias=True,
                                          dtype=dtype, device=device)

    def forward(
        self,
        tensor_src: torch.Tensor,
        tensor_tgt: torch.Tensor,
        depth_decoder_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tensor_src = 1.0 / tensor_src.clamp(min=1e-4)
        tensor_tgt = 1.0 / tensor_tgt.clamp(min=1e-4)
        tensor_input = torch.cat([tensor_src, tensor_tgt], dim=1)
        if self.depth_decoder_features:
            height, width = tensor_src.shape[-2:]
            upsampled_encodings = F.interpolate(
                depth_decoder_features, size=(height, width), mode="bilinear",
            )
            tensor_input = torch.cat([tensor_input, upsampled_encodings], dim=1)
        features = self.encoder(tensor_input)
        output = self.conv_out(self.decoder(features))
        alignment_map_lowres = self._act_fwd(output)
        if alignment_map_lowres.shape[-2:] != tensor_src.shape[-2:]:
            alignment_map = F.interpolate(
                alignment_map_lowres, size=tensor_src.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        else:
            alignment_map = alignment_map_lowres
        return alignment_map


# ===========================================================================
# Top-level model
# ===========================================================================


class DepthAlignment(nn.Module):
    """Depth alignment wrapper."""

    def __init__(self, scale_map_estimator: nn.Module | None):
        super().__init__()
        self.scale_map_estimator = scale_map_estimator

    def forward(
        self,
        monodepth: torch.Tensor,
        depth: torch.Tensor,
        depth_decoder_features: torch.Tensor | None = None,
    ):
        if depth is not None and self.scale_map_estimator is not None:
            depth_alignment_map = self.scale_map_estimator(
                monodepth[:, 0:1], depth, depth_decoder_features
            )
            monodepth = depth_alignment_map * monodepth
        else:
            depth_alignment_map = torch.ones_like(monodepth)
        return monodepth, depth_alignment_map


class RGBGaussianPredictor(nn.Module):
    """Predicts 3D Gaussians from images."""

    feature_model: nn.Module

    def __init__(
        self,
        init_model: nn.Module,
        monodepth_model: MonodepthWithEncodingAdaptor,
        feature_model: nn.Module,
        prediction_head: nn.Module,
        gaussian_composer: GaussianComposer,
        scale_map_estimator: nn.Module | None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.init_model = init_model
        self.feature_model = feature_model
        self.monodepth_model = monodepth_model
        self.prediction_head = prediction_head
        self.gaussian_composer = gaussian_composer
        self.depth_alignment = DepthAlignment(scale_map_estimator)

    def memory_required(self, input_shape):
        """Estimate peak activation VRAM (delegates to SPN encoder)."""
        return self.monodepth_model.monodepth_predictor.encoder.memory_required(input_shape)

    def encode(self, image: torch.Tensor):
        monodepth_output = self.monodepth_model(image)
        return monodepth_output, image

    def decode(
        self,
        monodepth_output,
        image: torch.Tensor,
        disparity_factor: torch.Tensor,
        depth: torch.Tensor | None = None,
    ) -> Gaussians3D:
        _dev = image.device
        _vram = lambda: torch.cuda.memory_allocated(_dev) / (1024 ** 3)
        log.warning(f"[Decode] start: {_vram():.2f} GB")
        monodepth_disparity = monodepth_output.disparity
        disparity_factor = disparity_factor[:, None, None, None]
        monodepth = disparity_factor / monodepth_disparity.clamp(min=1e-4, max=1e4)

        monodepth, _ = self.depth_alignment(
            monodepth, depth, monodepth_output.decoder_features,
        )
        log.warning(f"[Decode] after depth_alignment: {_vram():.2f} GB")

        init_output = self.init_model(image, monodepth)
        log.warning(f"[Decode] after init_model: {_vram():.2f} GB")
        image_features = self.feature_model(
            init_output.feature_input, encodings=monodepth_output.output_features
        )
        log.warning(f"[Decode] after feature_model: {_vram():.2f} GB")
        delta_values = self.prediction_head(image_features)
        log.warning(f"[Decode] after prediction_head: {_vram():.2f} GB")
        return self.gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )

    def forward(
        self,
        image: torch.Tensor,
        disparity_factor: torch.Tensor,
        depth: torch.Tensor | None = None,
    ) -> Gaussians3D:
        monodepth_output = self.monodepth_model(image)
        monodepth_disparity = monodepth_output.disparity

        disparity_factor = disparity_factor[:, None, None, None]
        monodepth = disparity_factor / monodepth_disparity.clamp(min=1e-4, max=1e4)

        monodepth, _ = self.depth_alignment(
            monodepth, depth, monodepth_output.decoder_features,
        )

        init_output = self.init_model(image, monodepth)
        image_features = self.feature_model(
            init_output.feature_input, encodings=monodepth_output.output_features
        )
        delta_values = self.prediction_head(image_features)
        return self.gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )

    def internal_resolution(self) -> int:
        return self.monodepth_model.internal_resolution()

    @property
    def output_resolution(self) -> int:
        return self.internal_resolution() // 2
