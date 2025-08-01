# Copyright (c) 2023-2024, Zexin He
#
# This code is adapted from https://github.com/3DTopia/OpenLRM
# with modifications to run openlrm on mindspore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mindspore import nn


class CameraEmbedder(nn.Cell):
    """
    Embed camera features to a high-dimensional vector.

    Reference:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L27
    """

    def __init__(self, raw_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.SequentialCell(
            nn.Dense(raw_dim, embed_dim),
            nn.SiLU(),
            nn.Dense(embed_dim, embed_dim),
        )

    def construct(self, x):
        return self.mlp(x)
