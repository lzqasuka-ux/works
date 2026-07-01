"""
Classic 3D Swin Transformer for sMRI Brain Image Binary Classification.

Reference: "Video Swin Transformer" (CVPR 2022) adapted to 3D medical imaging.
Input: 3D sMRI volume of shape (B, 1, D, H, W)
Output: Binary classification logits of shape (B, 2)

NOTE: Certain 3D-specific components (relative position bias table indexing,
shifted-window attention mask) are marked with TODO for later implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from itertools import product


# ==============================================================================
# Basic Building Blocks
# ==============================================================================

class Mlp(nn.Module):
    """Two-layer MLP with GELU activation and Dropout."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ==============================================================================
# 3D Patch Embedding
# ==============================================================================

class PatchEmbed3D(nn.Module):
    """
    Partition a 3D volume into non-overlapping patches and project to embedding space.

    Input:  (B, in_chans, D, H, W)
    Output: (B, num_patches, embed_dim)
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        in_chans: int = 1,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        # Patchify via 3D convolution
        x = self.proj(x)          # (B, embed_dim, D//pD, H//pH, W//pW)
        # Flatten spatial dims and transpose
        x = x.flatten(2)          # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


# ==============================================================================
# 3D Patch Merging (Downsample)
# ==============================================================================

class PatchMerging3D(nn.Module):
    """
    Merge 2x2x2 neighboring patches: spatial resolution halved, channels doubled.

    Input:  (B, D*H*W, C)
    Output: (B, (D/2)*(H/2)*(W/2), 2*C)
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int, int],
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        x = x.view(B, D, H, W, C)

        # Pad if spatial dims are odd
        pad_d = (2 - D % 2) % 2
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))

        D2, H2, W2 = D + pad_d, H + pad_h, W + pad_w
        D_out, H_out, W_out = D2 // 2, H2 // 2, W2 // 2

        # Gather 2x2x2 patches: (B, D_out, 2, H_out, 2, W_out, 2, C)
        x = x.view(B, D_out, 2, H_out, 2, W_out, 2, C)
        # Permute to group the 8 neighbors: (B, D_out, H_out, W_out, 2, 2, 2, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x = x.view(B, D_out * H_out * W_out, 8 * C)

        x = self.norm(x)
        x = self.reduction(x)  # (B, D_out*H_out*W_out, 2*C)
        return x


# ==============================================================================
# 3D Window Attention
# ==============================================================================

class WindowAttention3D(nn.Module):
    """
    3D Window-based Multi-head Self Attention (W-MSA / SW-MSA) with relative position bias.

    Supports shifted window via optional attention mask.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wd, Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define a parameter table of relative position bias
        # Shape: ((2*Wd-1) * (2*Wh-1) * (2*Ww-1), num_heads)
        total_window_elements = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(total_window_elements, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # TODO: Build the 3D relative position index table.
        # This table maps each pair of tokens within a 3D window to an index
        # in self.relative_position_bias_table.
        #
        # Expected shape: (window_volume, window_volume)
        # where window_volume = Wd * Wh * Ww.
        #
        # Steps:
        #   1. Create coordinate grids for [0..Wd-1], [0..Wh-1], [0..Ww-1].
        #   2. Compute pairwise relative coords: rel_d = coords_d[:, None] - coords_d[None, :]
        #      (same for h and w). Shape of each: (window_volume, window_volume).
        #   3. Shift to non-negative: rel_d += Wd - 1; rel_h += Wh - 1; rel_w += Ww - 1.
        #   4. Flatten to 1D index:
        #         index = rel_d * (2*Wh-1)*(2*Ww-1) + rel_h * (2*Ww-1) + rel_w
        #      Shape: (window_volume, window_volume).
        #   5. Register as non-trainable buffer: self.register_buffer("relative_position_index", index)
        #
        # self.relative_position_index = ...  # TODO: fill me
        self.relative_position_index: Optional[torch.Tensor] = None  # TODO

        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (num_windows*B, window_volume, C)
            mask: (num_windows, window_volume, window_volume) or None
                  → 3D shifted-window attention mask (TODO).
        Returns:
            (num_windows*B, window_volume, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x)                                    # (B_, N, 3*C)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)                    # (3, B_, num_heads, N, C_per_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)                       # (B_, num_heads, N, N)

        # Relative position bias
        if self.relative_position_index is not None:
            # TODO: Once self.relative_position_index is filled,
            # uncomment the lines below to add relative position bias.
            #
            # relative_position_bias = self.relative_position_bias_table[
            #     self.relative_position_index.view(-1)
            # ].view(N, N, -1)            # (N, N, num_heads)
            # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            # # → (num_heads, N, N), unsqueeze to (1, num_heads, N, N)
            # attn = attn + relative_position_bias.unsqueeze(0)
            pass

        # Attention mask for shifted windows
        # TODO: Apply 3D attention mask here once _build_3d_attention_mask is implemented.
        # nW = mask.shape[0] if mask is not None else 0
        # if mask is not None:
        #     attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _build_3d_attention_mask(
    shift_size: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    input_resolution: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Build attention mask for 3D cyclic-shifted window attention.

    When using shifted windows, tokens from different original windows
    should NOT attend to each other. This function builds a mask to enforce that.

    Args:
        shift_size: (sd, sh, sw) — shift amounts in each dimension.
        window_size: (wd, wh, ww) — window size.
        input_resolution: (D, H, W) — spatial resolution after patch embedding
                          (i.e., the grid size, NOT the original volume).

    Returns:
        attn_mask: (nW, window_volume, window_volume)
                   where nW = number of windows, window_volume = wd*wh*ww.
                   Values: 0 for allowed attention, -100.0 (or large negative) for masked.
    """
    # TODO: Implement 3D cyclic-shift attention mask.
    # The logic mirrors the 2D version in the original Swin Transformer paper
    # but extended to 3 dimensions.
    #
    # Outline:
    #   1. Compute img_mask of shape (1, D, H, W, 1) where each (d,h,w) cell
    #      gets a unique integer ID.
    #   2. Partition img_mask into windows: (nW, wd, wh, ww, 1) → (nW, window_volume).
    #   3. For each window, compute mask by comparing IDs:
    #         mask = (window_ids.unsqueeze(1) != window_ids.unsqueeze(2))
    #      → (nW, window_volume, window_volume).
    #   4. Fill True positions with a large negative value (e.g., -100.0).
    #
    # Placeholder:
    # return torch.zeros(1, window_volume, window_volume)  # TODO
    raise NotImplementedError(
        "_build_3d_attention_mask is not yet implemented. "
        "Fill this function following the outline above."
    )


# ==============================================================================
# 3D Swin Transformer Block
# ==============================================================================

class SwinTransformerBlock3D(nn.Module):
    """
    3D Swin Transformer Block.

    Consists of two consecutive sub-blocks:
      - Block 1: W-MSA  (regular window) + MLP
      - Block 2: SW-MSA (shifted window) + MLP
    Each sub-block has: Norm → Attention → Residual → Norm → MLP → Residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        shift_size: Tuple[int, int, int],
        input_resolution: Tuple[int, int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        # If window size equals input resolution, don't apply shift
        if min(input_resolution) <= min(window_size):
            self.shift_size = (0, 0, 0)
            self.window_size = (
                min(window_size[0], input_resolution[0]),
                min(window_size[1], input_resolution[1]),
                min(window_size[2], input_resolution[2]),
            )

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must be in [0, window_size)"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must be in [0, window_size)"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must be in [0, window_size)"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )

        # Build attention mask for SW-MSA block if shift > 0
        # TODO: When _build_3d_attention_mask is ready, uncomment:
        # if any(s > 0 for s in self.shift_size):
        #     self.register_buffer(
        #         "attn_mask",
        #         _build_3d_attention_mask(self.shift_size, self.window_size, self.input_resolution)
        #     )
        # else:
        #     self.attn_mask = None
        self.attn_mask: Optional[torch.Tensor] = None  # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # Cyclic shift
        sd, sh, sw = self.shift_size
        if sd > 0 or sh > 0 or sw > 0:
            x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(1, 2, 3))

        # Partition windows
        wd, wh, ww = self.window_size
        x_windows = _window_partition_3d(x, wd, wh, ww)
        # → (num_windows * B, wd, wh, ww, C)
        nW = x_windows.shape[0] // B
        x_windows = x_windows.view(-1, wd * wh * ww, C)
        # → (num_windows * B, window_volume, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        # → (num_windows * B, window_volume, C)

        # Merge windows back
        attn_windows = attn_windows.view(-1, wd, wh, ww, C)
        x = _window_reverse_3d(attn_windows, wd, wh, ww, D, H, W, B)
        # → (B, D, H, W, C)

        # Reverse cyclic shift
        if sd > 0 or sh > 0 or sw > 0:
            x = torch.roll(x, shifts=(sd, sh, sw), dims=(1, 2, 3))

        x = x.view(B, D * H * W, C)

        # FFN
        x = shortcut + x
        x_ffn = self.norm2(x)
        x_ffn = self.mlp(x_ffn)
        x = x + x_ffn

        return x


def _window_partition_3d(
    x: torch.Tensor,
    window_d: int,
    window_h: int,
    window_w: int
) -> torch.Tensor:
    """
    Partition a 3D feature map into non-overlapping windows.

    Args:
        x: (B, D, H, W, C)
        window_d, window_h, window_w: Window size in each dimension.

    Returns:
        Windows: (num_windows * B, window_d, window_h, window_w, C)
    """
    B, D, H, W, C = x.shape
    # Reshape: split each spatial dim into (num_splits, window_size)
    x = x.view(
        B,
        D // window_d, window_d,
        H // window_h, window_h,
        W // window_w, window_w,
        C
    )
    # Permute to bring window chunks together
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    # → (B, D//wd, H//wh, W//ww, wd, wh, ww, C)
    x = x.view(-1, window_d, window_h, window_w, C)
    return x


def _window_reverse_3d(
    windows: torch.Tensor,
    window_d: int,
    window_h: int,
    window_w: int,
    D: int,
    H: int,
    W: int,
    B: int
) -> torch.Tensor:
    """
    Reverse window partition back to a full 3D feature map.

    Args:
        windows: (num_windows * B, window_d, window_h, window_w, C)
        B: Batch size.

    Returns:
        Volume: (B, D, H, W, C)
    """
    C = windows.shape[-1]
    x = windows.view(
        B,
        D // window_d, H // window_h, W // window_w,
        window_d, window_h, window_w,
        C,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, C)
    return x


# ==============================================================================
# Basic Layer (one stage of Swin Transformer)
# ==============================================================================

class BasicLayer3D(nn.Module):
    """
    A Swin Transformer layer for one stage.

    Contains:
      - (Optional) PatchMerging3D for downsampling (except Stage 1)
      - Several SwinTransformerBlock3D blocks
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        input_resolution: Tuple[int, int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (
                    window_size[0] // 2,
                    window_size[1] // 2,
                    window_size[2] // 2,
                ),
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
            )
            for i in range(depth)
        ])

        # Patch merging layer (downsample after blocks)
        if downsample is not None:
            self.downsample = downsample(
                dim=dim,
                input_resolution=input_resolution,
                norm_layer=nn.LayerNorm,
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ==============================================================================
# 3D Swin Transformer Backbone
# ==============================================================================

class SwinTransformer3D(nn.Module):
    """
    3D Swin Transformer Backbone.

    Consists of:
      - PatchEmbed3D
      - 4 × BasicLayer3D (Stage 1~4)
      - LayerNorm output

    Returns feature maps from the last stage, suitable for classification
    or downstream task heads.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        in_chans: int = 1,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: Tuple[int, int, int] = (7, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        input_resolution: Optional[Tuple[int, int, int]] = None,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
    ):
        """
        Args:
            patch_size: 3D patch size for initial embedding.
            in_chans: Input channels (1 for grayscale MRI).
            embed_dim: Embedding dimension (C) for Stage 1.
            depths: Number of Swin blocks per stage.
            num_heads: Number of attention heads per stage.
            window_size: 3D window size.
            mlp_ratio: MLP hidden dimension ratio.
            qkv_bias: Whether to use bias in QKV projection.
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            input_resolution: (D, H, W) of the volume after patching.
                              If None, requires knowing volume size at forward.
            norm_layer: Normalization layer.
            patch_norm: Whether to apply LayerNorm after PatchEmbed.
            use_checkpoint: Whether to use gradient checkpointing.
        """
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # Build layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            resolution = (
                input_resolution[0] // (2 ** i_layer),
                input_resolution[1] // (2 ** i_layer),
                input_resolution[2] // (2 ** i_layer),
            ) if input_resolution is not None else None
            layer = BasicLayer3D(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                input_resolution=resolution,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                downsample=PatchMerging3D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) — 3D sMRI volume.

        Returns:
            (B, D_out * H_out * W_out, num_features) — flattened feature tokens.
        """
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)         # (B, num_patches_final, num_features)
        return x


# ==============================================================================
# Classification Model (Backbone + Head → Binary Classification)
# ==============================================================================

class sMRI3DSwinTransformer(nn.Module):
    """
    3D Swin Transformer for sMRI binary classification.

    Wraps the SwinTransformer3D backbone with a classification head:
      Backbone → AdaptiveAvgPool3d → Flatten → LayerNorm → Linear(num_features, num_classes)

    Input:  (B, 1, D, H, W)  — 3D sMRI brain volume
    Output: (B, num_classes)  — classification logits (default: 2 for binary)
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_chans: int = 1,
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: Tuple[int, int, int] = (7, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        input_resolution: Tuple[int, int, int] = (96, 96, 96),
        use_checkpoint: bool = False,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of output classes (2 for binary classification).
            in_chans: Number of input channels (1 for grayscale sMRI).
            patch_size: 3D patch size for tokenization (default (4,4,4)).
            embed_dim: Embedding dimension (C) in Stage 1.
            depths: Number of Swin Transformer blocks per stage.
            num_heads: Number of attention heads per stage.
            window_size: 3D window size for W-MSA.
            mlp_ratio: Hidden dimension expansion ratio for MLP.
            qkv_bias: Bias in QKV linear projection.
            drop_rate: Dropout rate after MLP / projection.
            attn_drop_rate: Dropout rate in attention weights.
            drop_path_rate: Stochastic depth decay rate.
            input_resolution: Expected volume shape (D, H, W) before patching.
            use_checkpoint: Use gradient checkpointing to save memory.
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_resolution = input_resolution

        # Backbone
        self.backbone = SwinTransformer3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            input_resolution=input_resolution,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=use_checkpoint,
        )

        num_features = self.backbone.num_features

        # Classification head
        self.head_norm = nn.LayerNorm(num_features)
        self.head = nn.Linear(num_features, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize head weights."""
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) — 3D sMRI volume.

        Returns:
            logits: (B, num_classes) — raw classification scores.
        """
        x = self.backbone(x)                         # (B, N, num_features)
        x = x.mean(dim=1)                            # (B, num_features) — global average pooling
        x = self.head_norm(x)
        x = self.head(x)                             # (B, num_classes)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classification head (useful for downstream tasks).

        Args:
            x: (B, C, D, H, W)

        Returns:
            features: (B, num_features)
        """
        x = self.backbone(x)
        x = x.mean(dim=1)
        return x