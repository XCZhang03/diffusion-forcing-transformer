from typing import Optional, Tuple, List
from functools import partial
from omegaconf import DictConfig
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from ..modules.embeddings import RotaryEmbedding3D, RandomDropoutPatchEmbed
from ..dit.dit_base import SinusoidalPositionalEmbedding
from .u_vit_blocks import (
    EmbedInput,
    ProjectOutput,
    ResBlock,
    TransformerBlock,
    Upsample,
    Downsample,
    AxialRotaryEmbedding,
    zero_module,
    Conv1x1AsLinear,
)
from .u_vit3d import UViT3D



class UViTControlBlock(UViT3D):
    """
    A U-ViT backbone from the following papers:
    - Simple diffusion: End-to-end diffusion for high resolution images (https://arxiv.org/abs/2301.11093)
    - Simpler Diffusion (SiD2): 1.5 FID on ImageNet512 with pixel-space diffusion (https://arxiv.org/abs/2410.19324)
    - We more closely follow SiD2's Residual U-ViT, where blockwise skip-connections are removed, and only a single skip-connection is used per downsampling operation.
    """

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        use_causal_mask=True,
    ):
        # ------------------------------- Configuration --------------------------------
        # these configurations closely follow the notation in the SiD2 paper
        channels = cfg.channels
        self.emb_dim = cfg.emb_channels
        self.input_embed_dim = channels[0]
        patch_size = cfg.patch_size
        block_types = cfg.block_types
        block_dropouts = cfg.block_dropouts
        num_updown_blocks = cfg.num_updown_blocks
        num_mid_blocks = cfg.num_mid_blocks
        num_heads = cfg.num_heads
        self.pos_emb_type = cfg.pos_emb_type
        self.num_levels = len(channels)
        resolution = x_shape[-1]
        self.is_transformers = [block_type != "ResBlock" for block_type in block_types]
        self.use_checkpointing = list(cfg.use_checkpointing)
        self.temporal_length = max_tokens
        self.conditioning_scale = cfg.get("conditioning_scale", 1.0)
        self.conditioning_dropout = cfg.external_cond_dropout

        # ------------------------------ Initialization ---------------------------------

        super(UViT3D, self).__init__(
            cfg,
            x_shape,
            max_tokens,
            external_cond_dim,
            use_causal_mask,
        )

        # -------------- Initial downsampling and final upsampling layers --------------
        # This enables avoiding high-resolution feature maps and speeds up the network
        # self.embed_input = EmbedInput(
        #     in_channels=x_shape[0],
        #     dim=channels[0],
        #     patch_size=patch_size,
        # )
        # self.input_projection = zero_module(nn.Conv2d(channels[0], channels[0]))

        # --------------------------- Positional embeddings ----------------------------
        # We use a 1D learnable positional embedding or RoPE for every level with transformers
        assert self.pos_emb_type in [
            "learned_1d",
            "rope",
        ], f"Positional embedding type {self.pos_emb_type} not supported."

        self.pos_embs = nn.ModuleDict({})
        for i_level, channel in enumerate(channels):
            if not self.is_transformers[i_level]:
                continue
            pos_emb_cls, dim = None, None
            if self.pos_emb_type == "rope":
                pos_emb_cls = (
                    RotaryEmbedding3D
                    if block_types[i_level] == "TransformerBlock"
                    else AxialRotaryEmbedding
                )
                dim = channel // num_heads
            else:
                pos_emb_cls = partial(SinusoidalPositionalEmbedding, learnable=True)
                dim = channel
            level_resolution = resolution // patch_size // (2**i_level)
            self.pos_embs[f"{i_level}"] = pos_emb_cls(
                dim,
                (self.temporal_length, level_resolution, level_resolution),
            )

        def _rope_kwargs(i_level: int):
            return (
                {"rope": self.pos_embs[f"{i_level}"]}
                if self.pos_emb_type == "rope" and self.is_transformers[i_level]
                else {}
            )

        self.down_blocks = nn.ModuleList()
        self.control_projections = nn.ModuleList()

        block_type_to_cls = {
            "ResBlock": partial(ResBlock, emb_dim=self.emb_dim),
            "TransformerBlock": partial(
                TransformerBlock, emb_dim=self.emb_dim, heads=num_heads
            ),
            "AxialTransformerBlock": partial(
                TransformerBlock,
                emb_dim=self.emb_dim,
                heads=num_heads,
                use_axial=True,
                ax1_len=self.temporal_length,
            ),
        }

        # ---------------------------- Down-sampling blocks ----------------------------
        for i_level, (num_blocks, ch, block_type, block_dropout) in enumerate(
            zip(
                num_updown_blocks,
                channels[:-1],
                block_types[:-1],
                block_dropouts[:-1],
            )
        ):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        block_type_to_cls[block_type](
                            ch, dropout=block_dropout, **_rope_kwargs(i_level)
                        )
                        for _ in range(num_blocks)
                    ]
                    + [Downsample(ch, channels[i_level + 1])],
                )
            )
            self.control_projections.append(
                zero_module(Conv1x1AsLinear(ch, ch, bias=True))
            )

        self.control_projections.append(
            zero_module(Conv1x1AsLinear(channels[-1], channels[-1], bias=True))
            )
        
        # ------------------------------ Middle blocks ---------------------------------
        self.mid_blocks = nn.ModuleList(
            [
                block_type_to_cls[block_types[-1]](
                    channels[-1],
                    dropout=block_dropouts[-1],
                    **_rope_kwargs(self.num_levels - 1),
                )
                for _ in range(num_mid_blocks // 2)
            ]
        )
        self.mid_control_projections = nn.ModuleList(
            [
                zero_module(Conv1x1AsLinear(channels[-1], channels[-1], bias=True)) if not self.is_transformers[-1] \
                else zero_module(nn.Linear(channels[-1], channels[-1], bias=True))
                for _ in range(num_mid_blocks // 2)
            ]
        )

    def _build_external_cond_embedding(self) -> Optional[nn.Module]:
        return zero_module(
            RandomDropoutPatchEmbed(
            dropout_prob=self.conditioning_dropout,
            img_size=self.x_shape[1],
            patch_size=self.cfg.patch_size,
            in_chans=self.external_cond_dim,
            embed_dim=self.external_cond_emb_dim,
            bias=True,
            flatten=False,
        )
        )

    @property
    def external_cond_emb_dim(self) -> int:
        return self.input_embed_dim

    def forward(
        self,
        noisy_image: Tensor,
        noise_levels: Tensor,
        external_cond: Optional[Tensor] = None,
        external_cond_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the U-ViT backbone.
        Args:
            noisy_image: Input tensor of shape (B, T, C, H, W).
            noise_levels: Noise level tensor of shape (B, T).
            external_cond: External pose tensor of shape (B, T, C, H, W).
        Returns:
            Output tensor of shape (B, T, C, H, W).
        """
        external_cond = self.external_cond_embedding(external_cond, external_cond_mask)
        external_cond = rearrange(external_cond, "b t c h w -> (b t) c h w")
        assert (
            external_cond.shape == noisy_image.shape
        ), f"pose condition should have the same shape as input, but got {external_cond.shape} instead of {x.shape}."

        x = noisy_image + external_cond

        # Embeddings
        emb = self.noise_level_pos_embedding(noise_levels)
        emb = rearrange(emb, "b t c -> (b t) c")

        hs_before = []  # hidden states before downsampling
        hs_after = []  # hidden states after downsampling
        hs_middle = [] # hidden states in mid block

        # Down-sampling blocks
        for i_level, down_block in enumerate(
            self.down_blocks,
        ):
            x = self._run_level(x, emb, i_level)
            hs_before.append(self.control_projections[i_level](x))
            x = down_block[-1](x)
            hs_after.append(self.control_projections[i_level + 1](x))

        # Middle blocks
        x, emb = self._rearrange_and_add_pos_emb_if_transformer(x, emb, self.num_levels - 1)
        for i_block, mid_block in enumerate(
            self.mid_blocks,
        ):
            x = self._checkpointed_forward(
                mid_block,
                x,
                emb,
                use_checkpointing=self.use_checkpointing[-1],
            )
            hs_middle.append(self.mid_control_projections[i_block](x))
        x = self._unrearrange_if_transformer(x, self.num_levels - 1)

        # scale
        hs_before = [hs * self.conditioning_scale for hs in hs_before]
        hs_after = [hs * self.conditioning_scale for hs in hs_after]
        hs_middle = [hs * self.conditioning_scale for hs in hs_middle]

        return hs_before, hs_after, hs_middle


class UViT3DControl(UViT3D):
    """
    U-ViT with control net.
    """

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        use_causal_mask=True,
    ):
        self.conditioning_dropout = cfg.external_cond_dropout
        super().__init__(
            cfg,
            x_shape,
            max_tokens,
            cfg.conditioning.dim,
            use_causal_mask,
        )
        self.control_net = UViTControlBlock(
            cfg,
            x_shape,
            max_tokens,
            external_cond_dim=x_shape[0],  ## pose condition has same shape as x
            use_causal_mask=use_causal_mask,
        )

        # No runtime grad hooks; use Conv1x1AsLinear for 1x1 convs to keep grads contiguous

    def _run_middle_block(self, x: Tensor, emb: Tensor, i_level: int, residuals: List[Tensor] = None) -> Tensor:
        """
        Run the blocks (except up/downsampling blocks) for a given level.
        Gradient checkpointing is used optionally, with self.checkpoints[i_level] segments.
        """
        use_checkpointing = self.use_checkpointing[i_level]
        blocks = self.mid_blocks
        for i, block in enumerate(blocks):
            x = self._checkpointed_forward(
                block,
                x,
                emb,
                use_checkpointing=use_checkpointing,
            )
            if i >= len(blocks) // 2:
                x += residuals.pop()
        return x

    def _run_level(
        self, x: Tensor, emb: Tensor, i_level: int, is_up: bool = False, residuals: List[Tensor] = None
    ) -> Tensor:
        """
        Run the blocks (except up/downsampling blocks) for a given level, accompanied by reshaping operations before and after.
        """
        x, emb = self._rearrange_and_add_pos_emb_if_transformer(x, emb, i_level)
        x = self._run_level_blocks(x, emb, i_level, is_up) if i_level != self.num_levels - 1 else self._run_middle_block(x, emb, i_level, residuals)
        x = self._unrearrange_if_transformer(x, i_level)
        return x

    def forward(
        self,
        x: Tensor,
        noise_levels: Tensor,
        external_cond: Optional[Tensor] = None,
        external_cond_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the U-ViT backbone, with pose conditioning.
        Args:
            x: Input tensor of shape (B, T, C, H, W).
            noise_levels: Noise level tensor of shape (B, T).
            external_cond: External conditioning tensor of shape (B, T, C', H, W).
        Returns:
            Output tensor of shape (B, T, C, H, W).
        """
        assert (
            x.shape[1] == self.temporal_length
        ), f"Temporal length of U-ViT is set to {self.temporal_length}, but input has temporal length {x.shape[1]}."

        assert (
            external_cond is not None
        ), "External condition (robot pose) is required for U-ViT3DControl model."

        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed_input(x)
        control_hs_before, control_hs_after, control_hs_middle = self.control_net(
            x,
            noise_levels,
            external_cond,
            external_cond_mask
        )

        # Embeddings
        emb = self.noise_level_pos_embedding(noise_levels)
        emb = rearrange(emb, "b t c -> (b t) c")

        hs_before = []  # hidden states before downsampling
        hs_after = []  # hidden states after downsampling

        # Down-sampling blocks
        for i_level, down_block in enumerate(
            self.down_blocks,
        ):
            x = self._run_level(x, emb, i_level)
            hs_before.append(x)
            x = down_block[-1](x)
            hs_after.append(x)

        # Middle blocks
        x = self._run_level(x, emb, self.num_levels - 1, residuals=control_hs_middle)

        # add control net output
        assert len(hs_before) == len(control_hs_before)
        assert len(hs_after) == len(control_hs_after)
        for i_level in range(len(hs_before)):
            hs_before[i_level] = hs_before[i_level] + control_hs_before[i_level]
        for i_level in range(len(hs_after)):
            hs_after[i_level] = hs_after[i_level] + control_hs_after[i_level]

        # Up-sampling blocks
        for _i_level, up_block in enumerate(self.up_blocks):
            i_level = self.num_levels - 2 - _i_level
            x = x - hs_after.pop()
            x = up_block[0](x) + hs_before.pop()
            x = self._run_level(x, emb, i_level, is_up=True)

        x = self.project_output(x)
        return rearrange(x, "(b t) c h w -> b t c h w", t=self.temporal_length)
