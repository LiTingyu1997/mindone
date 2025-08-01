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
import mindspore as ms
from mindspore import mint, nn

from mindone.transformers import CLIPPreTrainedModel, CLIPVisionModel

from ...models.attention import BasicTransformerBlock
from ...utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PaintByExampleImageEncoder(CLIPPreTrainedModel):
    def __init__(self, config, proj_size=None):
        super().__init__(config)
        self.proj_size = proj_size or getattr(config, "projection_dim", 768)

        self.model = CLIPVisionModel(config)
        self.mapper = PaintByExampleMapper(config)
        self.final_layer_norm = mint.nn.LayerNorm(config.hidden_size)
        self.proj_out = mint.nn.Linear(config.hidden_size, self.proj_size)

        # uncondition for scaling
        self.uncond_vector = ms.Parameter(mint.randn((1, 1, self.proj_size)), name="uncond_vector")

    def construct(self, pixel_values, return_uncond_vector=False):
        clip_output = self.model(pixel_values=pixel_values)
        latent_states = clip_output[1]
        latent_states = self.mapper(latent_states[:, None])
        latent_states = self.final_layer_norm(latent_states)
        latent_states = self.proj_out(latent_states)
        if return_uncond_vector:
            return latent_states, self.uncond_vector

        return latent_states


class PaintByExampleMapper(nn.Cell):
    def __init__(self, config):
        super().__init__()
        num_layers = (config.num_hidden_layers + 1) // 5
        hid_size = config.hidden_size
        num_heads = 1
        self.blocks = nn.CellList(
            [
                BasicTransformerBlock(hid_size, num_heads, hid_size, activation_fn="gelu", attention_bias=True)
                for _ in range(num_layers)
            ]
        )

    def construct(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)

        return hidden_states
