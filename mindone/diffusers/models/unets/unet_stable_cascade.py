# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import mint, nn
from mindspore.common.initializer import Constant, Normal, XavierNormal, initializer

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin
from ...utils import BaseOutput
from ..attention_processor import Attention
from ..modeling_utils import ModelMixin


# Copied from diffusers.pipelines.wuerstchen.modeling_wuerstchen_common.WuerstchenLayerNorm with WuerstchenLayerNorm -> SDCascadeLayerNorm
class SDCascadeLayerNorm(mint.nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def construct(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().construct(x)
        return x.permute(0, 3, 1, 2)


class SDCascadeTimestepBlock(nn.Cell):
    def __init__(self, c, c_timestep, conds=[]):
        super().__init__()

        self.mapper = mint.nn.Linear(c_timestep, c * 2)
        self.conds = conds
        for cname in conds:
            setattr(self, f"mapper_{cname}", mint.nn.Linear(c_timestep, c * 2))

    def construct(self, x, t):
        t = t.chunk(len(self.conds) + 1, dim=1)
        a, b = self.mapper(t[0])[:, :, None, None].chunk(2, dim=1)
        for i, c in enumerate(self.conds):
            ac, bc = getattr(self, f"mapper_{c}")(t[i + 1])[:, :, None, None].chunk(2, dim=1)
            a, b = a + ac, b + bc
        return x * (1 + a) + b


class SDCascadeResBlock(nn.Cell):
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0):
        super().__init__()
        self.depthwise = mint.nn.Conv2d(c, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        self.norm = SDCascadeLayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.SequentialCell(
            mint.nn.Linear(c + c_skip, c * 4),
            mint.nn.GELU(),
            GlobalResponseNorm(c * 4),
            mint.nn.Dropout(p=dropout),
            mint.nn.Linear(c * 4, c),
        )

    def construct(self, x, x_skip=None):
        x_res = x
        x = self.norm(self.depthwise(x))
        if x_skip is not None:
            x = mint.cat([x, x_skip], dim=1)
        x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + x_res


# from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
class GlobalResponseNorm(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.gamma = ms.Parameter(mint.zeros((1, 1, 1, dim)), name="gamma")
        self.beta = ms.Parameter(mint.zeros((1, 1, 1, dim)), name="beta")

    def construct(self, x):
        agg_norm = mint.norm(x, p="fro", dim=(1, 2), keepdim=True).to(x.dtype)
        stand_div_norm = agg_norm / (agg_norm.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * stand_div_norm) + self.beta + x


class SDCascadeAttnBlock(nn.Cell):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        super().__init__()

        self.self_attn = self_attn
        self.norm = SDCascadeLayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(query_dim=c, heads=nhead, dim_head=c // nhead, dropout=dropout, bias=True)
        self.kv_mapper = nn.SequentialCell(mint.nn.SiLU(), mint.nn.Linear(c_cond, c))

    def construct(self, x, kv):
        kv = self.kv_mapper(kv)
        norm_x = self.norm(x)
        if self.self_attn:
            batch_size, channel, _, _ = x.shape
            kv = mint.cat([norm_x.view(batch_size, channel, -1).swapaxes(1, 2), kv], dim=1)
        x = x + self.attention(norm_x, encoder_hidden_states=kv)
        return x


class UpDownBlock2d(nn.Cell):
    def __init__(self, in_channels, out_channels, mode, enabled=True):
        super().__init__()
        if mode not in ["up", "down"]:
            raise ValueError(f"{mode} not supported")
        interpolation = (
            mint.nn.Upsample(scale_factor=2 if mode == "up" else 0.5, mode="bilinear", align_corners=True)
            if enabled
            else mint.nn.Identity()
        )
        mapping = mint.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.blocks = nn.CellList([interpolation, mapping] if mode == "up" else [mapping, interpolation])

    def construct(self, x):
        for block in self.blocks:
            x = block(x)
        return x


@dataclass
class StableCascadeUNetOutput(BaseOutput):
    sample: ms.Tensor = None


class StableCascadeUNet(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        timestep_ratio_embedding_dim: int = 64,
        patch_size: int = 1,
        conditioning_dim: int = 2048,
        block_out_channels: Tuple[int] = (2048, 2048),
        num_attention_heads: Tuple[int] = (32, 32),
        down_num_layers_per_block: Tuple[int] = (8, 24),
        up_num_layers_per_block: Tuple[int] = (24, 8),
        down_blocks_repeat_mappers: Optional[Tuple[int]] = (
            1,
            1,
        ),
        up_blocks_repeat_mappers: Optional[Tuple[int]] = (1, 1),
        block_types_per_layer: Tuple[Tuple[str]] = (
            ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
            ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
        ),
        clip_text_in_channels: Optional[int] = None,
        clip_text_pooled_in_channels=1280,
        clip_image_in_channels: Optional[int] = None,
        clip_seq=4,
        effnet_in_channels: Optional[int] = None,
        pixel_mapper_in_channels: Optional[int] = None,
        kernel_size=3,
        dropout: Union[float, Tuple[float]] = (0.1, 0.1),
        self_attn: Union[bool, Tuple[bool]] = True,
        timestep_conditioning_type: Tuple[str] = ("sca", "crp"),
        switch_level: Optional[Tuple[bool]] = None,
    ):
        """

        Parameters:
            in_channels (`int`, defaults to 16):
                Number of channels in the input sample.
            out_channels (`int`, defaults to 16):
                Number of channels in the output sample.
            timestep_ratio_embedding_dim (`int`, defaults to 64):
                Dimension of the projected time embedding.
            patch_size (`int`, defaults to 1):
                Patch size to use for pixel unshuffling layer
            conditioning_dim (`int`, defaults to 2048):
                Dimension of the image and text conditional embedding.
            block_out_channels (Tuple[int], defaults to (2048, 2048)):
                Tuple of output channels for each block.
            num_attention_heads (Tuple[int], defaults to (32, 32)):
                Number of attention heads in each attention block. Set to -1 to if block types in a layer do not have
                attention.
            down_num_layers_per_block (Tuple[int], defaults to [8, 24]):
                Number of layers in each down block.
            up_num_layers_per_block (Tuple[int], defaults to [24, 8]):
                Number of layers in each up block.
            down_blocks_repeat_mappers (Tuple[int], optional, defaults to [1, 1]):
                Number of 1x1 Convolutional layers to repeat in each down block.
            up_blocks_repeat_mappers (Tuple[int], optional, defaults to [1, 1]):
                Number of 1x1 Convolutional layers to repeat in each up block.
            block_types_per_layer (Tuple[Tuple[str]], optional,
                defaults to (
                    ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"), ("SDCascadeResBlock",
                    "SDCascadeTimestepBlock", "SDCascadeAttnBlock")
                ): Block types used in each layer of the up/down blocks.
            clip_text_in_channels (`int`, *optional*, defaults to `None`):
                Number of input channels for CLIP based text conditioning.
            clip_text_pooled_in_channels (`int`, *optional*, defaults to 1280):
                Number of input channels for pooled CLIP text embeddings.
            clip_image_in_channels (`int`, *optional*):
                Number of input channels for CLIP based image conditioning.
            clip_seq (`int`, *optional*, defaults to 4):
            effnet_in_channels (`int`, *optional*, defaults to `None`):
                Number of input channels for effnet conditioning.
            pixel_mapper_in_channels (`int`, defaults to `None`):
                Number of input channels for pixel mapper conditioning.
            kernel_size (`int`, *optional*, defaults to 3):
                Kernel size to use in the block convolutional layers.
            dropout (Tuple[float], *optional*, defaults to (0.1, 0.1)):
                Dropout to use per block.
            self_attn (Union[bool, Tuple[bool]]):
                Tuple of booleans that determine whether to use self attention in a block or not.
            timestep_conditioning_type (Tuple[str], defaults to ("sca", "crp")):
                Timestep conditioning type.
            switch_level (Optional[Tuple[bool]], *optional*, defaults to `None`):
                Tuple that indicates whether upsampling or downsampling should be applied in a block
        """

        super().__init__()

        if len(block_out_channels) != len(down_num_layers_per_block):
            raise ValueError(
                f"Number of elements in `down_num_layers_per_block` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        elif len(block_out_channels) != len(up_num_layers_per_block):
            raise ValueError(
                f"Number of elements in `up_num_layers_per_block` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        elif len(block_out_channels) != len(down_blocks_repeat_mappers):
            raise ValueError(
                f"Number of elements in `down_blocks_repeat_mappers` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        elif len(block_out_channels) != len(up_blocks_repeat_mappers):
            raise ValueError(
                f"Number of elements in `up_blocks_repeat_mappers` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        elif len(block_out_channels) != len(block_types_per_layer):
            raise ValueError(
                f"Number of elements in `block_types_per_layer` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        if isinstance(dropout, float):
            dropout = (dropout,) * len(block_out_channels)
        if isinstance(self_attn, bool):
            self_attn = (self_attn,) * len(block_out_channels)

        # CONDITIONING
        if effnet_in_channels is not None:
            self.effnet_mapper = nn.SequentialCell(
                mint.nn.Conv2d(effnet_in_channels, block_out_channels[0] * 4, kernel_size=1),
                mint.nn.GELU(),
                mint.nn.Conv2d(block_out_channels[0] * 4, block_out_channels[0], kernel_size=1),
                SDCascadeLayerNorm(block_out_channels[0], elementwise_affine=False, eps=1e-6),
            )
        else:
            self.effnet_mapper = None
        if pixel_mapper_in_channels is not None:
            self.pixels_mapper = nn.SequentialCell(
                mint.nn.Conv2d(pixel_mapper_in_channels, block_out_channels[0] * 4, kernel_size=1),
                mint.nn.GELU(),
                mint.nn.Conv2d(block_out_channels[0] * 4, block_out_channels[0], kernel_size=1),
                SDCascadeLayerNorm(block_out_channels[0], elementwise_affine=False, eps=1e-6),
            )
        else:
            self.pixels_mapper = None

        self.clip_txt_pooled_mapper = mint.nn.Linear(clip_text_pooled_in_channels, conditioning_dim * clip_seq)
        if clip_text_in_channels is not None:
            self.clip_txt_mapper = mint.nn.Linear(clip_text_in_channels, conditioning_dim)
        if clip_image_in_channels is not None:
            self.clip_img_mapper = mint.nn.Linear(clip_image_in_channels, conditioning_dim * clip_seq)
        self.clip_norm = mint.nn.LayerNorm(conditioning_dim, elementwise_affine=False, eps=1e-6)

        self.embedding = nn.SequentialCell(
            # todo: unavailable mint interface
            nn.PixelUnshuffle(patch_size),
            mint.nn.Conv2d(in_channels * (patch_size**2), block_out_channels[0], kernel_size=1),
            SDCascadeLayerNorm(block_out_channels[0], elementwise_affine=False, eps=1e-6),
        )

        def get_block(block_type, in_channels, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == "SDCascadeResBlock":
                return SDCascadeResBlock(in_channels, c_skip, kernel_size=kernel_size, dropout=dropout)
            elif block_type == "SDCascadeAttnBlock":
                return SDCascadeAttnBlock(in_channels, conditioning_dim, nhead, self_attn=self_attn, dropout=dropout)
            elif block_type == "SDCascadeTimestepBlock":
                return SDCascadeTimestepBlock(
                    in_channels, timestep_ratio_embedding_dim, conds=timestep_conditioning_type
                )
            else:
                raise ValueError(f"Block type {block_type} not supported")

        # BLOCKS
        # -- down blocks
        down_blocks = []
        down_downscalers = []
        down_repeat_mappers = []
        for i in range(len(block_out_channels)):
            if i > 0:
                down_downscalers.append(
                    nn.SequentialCell(
                        SDCascadeLayerNorm(block_out_channels[i - 1], elementwise_affine=False, eps=1e-6),
                        UpDownBlock2d(
                            block_out_channels[i - 1], block_out_channels[i], mode="down", enabled=switch_level[i - 1]
                        )
                        if switch_level is not None
                        else mint.nn.Conv2d(
                            block_out_channels[i - 1],
                            block_out_channels[i],
                            kernel_size=2,
                            stride=2,
                        ),
                    )
                )
            else:
                down_downscalers.append(mint.nn.Identity())

            down_block = []
            for _ in range(down_num_layers_per_block[i]):
                for block_type in block_types_per_layer[i]:
                    block = get_block(
                        block_type,
                        block_out_channels[i],
                        num_attention_heads[i],
                        dropout=dropout[i],
                        self_attn=self_attn[i],
                    )
                    down_block.append(block)
            down_blocks.append(nn.CellList(down_block))

            if down_blocks_repeat_mappers is not None:
                block_repeat_mappers = []
                for _ in range(down_blocks_repeat_mappers[i] - 1):
                    block_repeat_mappers.append(
                        mint.nn.Conv2d(block_out_channels[i], block_out_channels[i], kernel_size=1)
                    )
                down_repeat_mappers.append(nn.CellList(block_repeat_mappers))

        self.down_blocks = nn.CellList(down_blocks)
        self.down_downscalers = nn.CellList(down_downscalers)
        self.down_repeat_mappers = nn.CellList(down_repeat_mappers)

        # -- up blocks
        up_blocks = []
        up_upscalers = []
        up_repeat_mappers = []
        for i in reversed(range(len(block_out_channels))):
            if i > 0:
                up_upscalers.append(
                    nn.SequentialCell(
                        SDCascadeLayerNorm(block_out_channels[i], elementwise_affine=False, eps=1e-6),
                        UpDownBlock2d(
                            block_out_channels[i], block_out_channels[i - 1], mode="up", enabled=switch_level[i - 1]
                        )
                        if switch_level is not None
                        else mint.nn.ConvTranspose2d(
                            block_out_channels[i], block_out_channels[i - 1], kernel_size=2, stride=2
                        ),
                    )
                )
            else:
                up_upscalers.append(mint.nn.Identity())

            up_block = []
            for j in range(up_num_layers_per_block[::-1][i]):
                for k, block_type in enumerate(block_types_per_layer[i]):
                    c_skip = block_out_channels[i] if i < len(block_out_channels) - 1 and j == k == 0 else 0
                    block = get_block(
                        block_type,
                        block_out_channels[i],
                        num_attention_heads[i],
                        c_skip=c_skip,
                        dropout=dropout[i],
                        self_attn=self_attn[i],
                    )
                    up_block.append(block)
            up_blocks.append(nn.CellList(up_block))

            if up_blocks_repeat_mappers is not None:
                block_repeat_mappers = []
                for _ in range(up_blocks_repeat_mappers[::-1][i] - 1):
                    block_repeat_mappers.append(
                        mint.nn.Conv2d(block_out_channels[i], block_out_channels[i], kernel_size=1)
                    )
                up_repeat_mappers.append(nn.CellList(block_repeat_mappers))

        self.up_blocks = nn.CellList(up_blocks)
        self.up_upscalers = nn.CellList(up_upscalers)
        self.up_repeat_mappers = nn.CellList(up_repeat_mappers)

        # OUTPUT
        self.clf = nn.SequentialCell(
            SDCascadeLayerNorm(block_out_channels[0], elementwise_affine=False, eps=1e-6),
            mint.nn.Conv2d(block_out_channels[0], out_channels * (patch_size**2), kernel_size=1),
            # todo: unavailable mint interface
            nn.PixelShuffle(patch_size),
        )

        self.gradient_checkpointing = False

    def _init_weights(self, m):
        if isinstance(m, (mint.nn.Conv2d, mint.nn.Linear)):
            m.weight.set_data(initializer(XavierNormal(), m.weight.shape, m.weight.dtype))
            if m.bias is not None:
                m.bias.set_data(initializer(Constant(0), m.bias.shape, m.bias.dtype))

        self.clip_txt_pooled_mapper.weight.set_data(
            initializer(
                Normal(sigma=0.02), self.clip_txt_pooled_mapper.weight.shape, self.clip_txt_pooled_mapper.weight.dtype
            )
        )
        self.clip_txt_mapper.weight.set_data(
            initializer(Normal(sigma=0.02), self.clip_txt_mapper.weight.shape, self.clip_txt_mapper.weight.dtype)
        ) if hasattr(self, "clip_txt_mapper") else None
        self.clip_img_mapper.weight.set_data(
            initializer(Normal(sigma=0.02), self.clip_img_mapper.weight.shape, self.clip_img_mapper.weight.dtype)
        ) if hasattr(self, "clip_img_mapper") else None

        if hasattr(self, "effnet_mapper"):
            self.effnet_mapper[0].weight.set_data(
                initializer(Normal(sigma=0.02), self.effnet_mapper[0].weight.shape, self.effnet_mapper[0].weight.dtype)
            )  # conditionings
            self.effnet_mapper[2].weight.set_data(
                initializer(Normal(sigma=0.02), self.effnet_mapper[2].weight.shape, self.effnet_mapper[2].weight.dtype)
            )  # conditionings

        if hasattr(self, "pixels_mapper"):
            self.pixels_mapper[0].weight.set_data(
                initializer(Normal(sigma=0.02), self.pixels_mapper[0].weight.shape, self.pixels_mapper[0].weight.dtype)
            )  # conditionings
            self.pixels_mapper[2].weight.set_data(
                initializer(Normal(sigma=0.02), self.pixels_mapper[2].weight.shape, self.pixels_mapper[2].weight.dtype)
            )  # conditionings

        self.embedding[1].weight.set_data(
            initializer(XavierNormal(gain=0.02), self.embedding[1].weight.shape, self.embedding[1].weight.dtype)
        )  # inputs
        self.clf[1].weight.set_data(initializer(0, self.clf[1].weight.shape, self.clf[1].weight.dtype))  # outputs

        # blocks
        for level_block in self.down_blocks + self.up_blocks:
            for block in level_block:
                if isinstance(block, SDCascadeResBlock):
                    block.channelwise[-1].weight *= np.sqrt(1 / sum(self.config.blocks[0]))
                elif isinstance(block, SDCascadeTimestepBlock):
                    block.mapper.weight.data.constant_(0)

    def get_timestep_ratio_embedding(self, timestep_ratio, max_positions=10000):
        r = timestep_ratio * max_positions
        half_dim = self.config["timestep_ratio_embedding_dim"] // 2

        emb = math.log(max_positions) / (half_dim - 1)
        emb = mint.exp(mint.mul(mint.arange(half_dim).float(), -emb))
        emb = r[:, None] * emb[None, :]
        emb = mint.cat([emb.sin(), emb.cos()], dim=1)

        if self.config["timestep_ratio_embedding_dim"] % 2 == 1:  # zero pad
            emb = mint.nn.functional.pad(emb, (0, 1), mode="constant")

        return emb.to(dtype=r.dtype)

    def get_clip_embeddings(self, clip_txt_pooled, clip_txt=None, clip_img=None):
        if len(clip_txt_pooled.shape) == 2:
            clip_txt_pool = clip_txt_pooled.unsqueeze(1)
        clip_txt_pool = self.clip_txt_pooled_mapper(clip_txt_pooled).view(
            clip_txt_pooled.shape[0], clip_txt_pooled.shape[1] * self.config["clip_seq"], -1
        )
        if clip_txt is not None and clip_img is not None:
            clip_txt = self.clip_txt_mapper(clip_txt)
            if len(clip_img.shape) == 2:
                clip_img = clip_img.unsqueeze(1)
            clip_img = self.clip_img_mapper(clip_img).view(
                clip_img.shape[0], clip_img.shape[1] * self.config["clip_seq"], -1
            )
            clip = mint.cat([clip_txt, clip_txt_pool, clip_img], dim=1)
        else:
            clip = clip_txt_pool
        return self.clip_norm(clip)

    def _down_encode(self, x, r_embed, clip):
        level_outputs = []
        block_group = list(zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers))

        for down_block, downscaler, repmap in block_group:
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for block in down_block:
                    if isinstance(block, SDCascadeResBlock):
                        x = block(x)
                    elif isinstance(block, SDCascadeAttnBlock):
                        x = block(x, clip)
                    elif isinstance(block, SDCascadeTimestepBlock):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs = [x] + level_outputs
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, clip):
        x = level_outputs[0]
        block_group = list(zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers))

        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    if isinstance(block, SDCascadeResBlock):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]):
                            orig_type = x.dtype
                            x = mint.nn.functional.interpolate(
                                x.float(), skip.shape[-2:], mode="bilinear", align_corners=True
                            )
                            x = x.to(orig_type)
                        x = block(x, skip)
                    elif isinstance(block, SDCascadeAttnBlock):
                        x = block(x, clip)
                    elif isinstance(block, SDCascadeTimestepBlock):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)
        return x

    def construct(
        self,
        sample,
        timestep_ratio,
        clip_text_pooled,
        clip_text=None,
        clip_img=None,
        effnet=None,
        pixels=None,
        sca=None,
        crp=None,
        return_dict=False,
    ):
        if pixels is None:
            pixels = sample.new_zeros((sample.shape[0], 3, 8, 8), dtype=sample.dtype)

        # Process the conditioning embeddings
        timestep_ratio_embed = self.get_timestep_ratio_embedding(timestep_ratio)
        for c in self.config["timestep_conditioning_type"]:
            if c == "sca":
                cond = sca
            elif c == "crp":
                cond = crp
            else:
                cond = None
            t_cond = cond or mint.zeros_like(timestep_ratio)
            timestep_ratio_embed = mint.cat([timestep_ratio_embed, self.get_timestep_ratio_embedding(t_cond)], dim=1)
        clip = self.get_clip_embeddings(clip_txt_pooled=clip_text_pooled, clip_txt=clip_text, clip_img=clip_img)

        # Model Blocks
        x = self.embedding(sample)
        if self.effnet_mapper is not None and effnet is not None:
            x = x + self.effnet_mapper(
                mint.nn.functional.interpolate(effnet, size=x.shape[-2:], mode="bilinear", align_corners=True)
            )
        if self.pixels_mapper is not None:
            x = x + mint.nn.functional.interpolate(
                self.pixels_mapper(pixels), size=x.shape[-2:], mode="bilinear", align_corners=True
            )
        level_outputs = self._down_encode(x, timestep_ratio_embed, clip)
        x = self._up_decode(level_outputs, timestep_ratio_embed, clip)
        sample = self.clf(x)

        if not return_dict:
            return (sample,)
        return StableCascadeUNetOutput(sample=sample)
