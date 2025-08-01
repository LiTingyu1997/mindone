# coding=utf-8
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
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
"""MindSpore DeBERTa-v2 model."""

from collections.abc import Sequence
from typing import Optional, Tuple, Union

from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.utils import logging

import mindspore as ms
from mindspore import jit, mint, nn
from mindspore.common.initializer import Normal, One, Zero, initializer

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import dtype_to_min
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import MSPreTrainedModel

logger = logging.get_logger(__name__)


# Copied from transformers.models.deberta.modeling_deberta.DebertaSelfOutput with DebertaLayerNorm->LayerNorm
class DebertaV2SelfOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


@jit
def make_log_bucket_position(relative_pos, bucket_size: int, max_position: int):
    sign = mint.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = mint.where(
        (relative_pos < mid) & (relative_pos > -mid),
        ms.tensor(mid - 1).type_as(relative_pos),
        mint.abs(relative_pos),
    )
    log_pos = mint.ceil(mint.log(abs_pos / mid) / mint.log(ms.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
    bucket_pos = mint.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
    return bucket_pos


def build_relative_position(query_layer, key_layer, bucket_size: int = -1, max_position: int = -1):
    """
    Build relative position according to the query and key

    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)

    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maximum allowed absolute position

    Return:
        `ms.tensor`: A tensor with shape [1, query_size, key_size]
    """
    query_size = query_layer.shape[-2]
    key_size = key_layer.shape[-2]

    q_ids = mint.arange(query_size, dtype=ms.int32)
    k_ids = mint.arange(key_size, dtype=ms.int32)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.to(ms.int64)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


@jit
# Copied from transformers.models.deberta.modeling_deberta.c2p_dynamic_expand
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.broadcast_to(
        [query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], relative_pos.shape[-1]]
    )


@jit
# Copied from transformers.models.deberta.modeling_deberta.p2c_dynamic_expand
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.broadcast_to([query_layer.shape[0], query_layer.shape[1], key_layer.shape[2], key_layer.shape[-2]])


@jit
# Copied from transformers.models.deberta.modeling_deberta.pos_dynamic_expand
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.broadcast_to(p2c_att.shape[:2] + (pos_index.shape[-2], key_layer.shape[-2]))


@jit
def scaled_size_sqrt(query_layer: ms.Tensor, scale_factor: int):
    return mint.sqrt(ms.tensor(query_layer.shape[-1], dtype=ms.float32) * scale_factor)


@jit
def build_rpos(query_layer, key_layer, relative_pos, position_buckets: int, max_relative_positions: int):
    if key_layer.shape[-2] != query_layer.shape[-2]:
        return build_relative_position(
            key_layer,
            key_layer,
            bucket_size=position_buckets,
            max_position=max_relative_positions,
        )
    else:
        return relative_pos


class DisentangledSelfAttention(nn.Cell):
    """
    Disentangled self-attention module

    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]

    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(config, "attention_head_size", _attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = mint.nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = mint.nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = mint.nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = mint.nn.Dropout(config.hidden_dropout_prob)

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = mint.nn.Linear(config.hidden_size, self.all_head_size, bias=True)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = mint.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = mint.nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, attention_heads) -> ms.Tensor:
        new_x_shape = x.shape[:-1] + (attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.shape[1], x.shape[-1])

    def construct(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        """
        Call the module

        Args:
            hidden_states (`ms.Tensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                *Attention(Q,K,V)*

            attention_mask (`ms.Tensor`):
                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
                th token.

            output_attentions (`bool`, *optional*):
                Whether return the attention matrix.

            query_states (`ms.Tensor`, *optional*):
                The *Q* state in *Attention(Q,K,V)*.

            relative_pos (`ms.Tensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
                values ranging in [*-max_relative_positions*, *max_relative_positions*].

            rel_embeddings (`ms.Tensor`):
                The embedding of relative distances. It's a tensor of shape [\\(2 \\times
                \\text{max_relative_positions}\\), *hidden_size*].


        """
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = scaled_size_sqrt(query_layer, scale_factor)
        attention_scores = mint.bmm(query_layer, key_layer.transpose(0, 2, 1) / scale.to(dtype=query_layer.dtype))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = attention_scores
        attention_scores = attention_scores.view(
            -1, self.num_attention_heads, attention_scores.shape[-2], attention_scores.shape[-1]
        )

        attention_mask = attention_mask.bool()
        attention_scores = attention_scores.masked_fill(~(attention_mask), dtype_to_min(query_layer.dtype))
        # bsz x height x length x dimension
        attention_probs = mint.nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)
        context_layer = mint.bmm(
            attention_probs.view(-1, attention_probs.shape[-2], attention_probs.shape[-1]), value_layer
        )
        context_layer = (
            context_layer.view(-1, self.num_attention_heads, context_layer.shape[-2], context_layer.shape[-1])
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        new_context_layer_shape = context_layer.shape[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)
        if not output_attentions:
            return (context_layer, None)
        return (context_layer, attention_probs)

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            relative_pos = build_relative_position(
                query_layer,
                key_layer,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bsz x height x query x key
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long()

        rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(
                self.query_proj(rel_embeddings), self.num_attention_heads
            ).repeat(query_layer.shape[0] // self.num_attention_heads, 1, 1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads).repeat(
                query_layer.shape[0] // self.num_attention_heads, 1, 1
            )
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.pos_key_proj(rel_embeddings), self.num_attention_heads
                ).repeat(
                    query_layer.shape[0] // self.num_attention_heads, 1, 1
                )  # .split(self.all_head_size, dim=-1)
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.pos_query_proj(rel_embeddings), self.num_attention_heads
                ).repeat(
                    query_layer.shape[0] // self.num_attention_heads, 1, 1
                )  # .split(self.all_head_size, dim=-1)

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = scaled_size_sqrt(pos_key_layer, scale_factor)
            c2p_att = mint.bmm(query_layer, pos_key_layer.transpose(0, 2, 1))
            c2p_pos = mint.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = mint.gather(
                c2p_att,
                dim=-1,
                index=c2p_pos.squeeze(0).broadcast_to(
                    [query_layer.shape[0], query_layer.shape[1], relative_pos.shape[-1]]
                ),
            )
            score += c2p_att / scale.to(dtype=c2p_att.dtype)

        # position->content
        if "p2c" in self.pos_att_type:
            scale = scaled_size_sqrt(pos_query_layer, scale_factor)
            r_pos = build_rpos(
                query_layer,
                key_layer,
                relative_pos,
                self.max_relative_positions,
                self.position_buckets,
            )
            p2c_pos = mint.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = mint.bmm(key_layer, pos_query_layer.transpose(0, 2, 1))
            p2c_att = mint.gather(
                p2c_att,
                dim=-1,
                index=p2c_pos.squeeze(0).broadcast_to([query_layer.shape[0], key_layer.shape[-2], key_layer.shape[-2]]),
            ).transpose(0, 2, 1)
            score += p2c_att / scale.to(dtype=p2c_att.dtype)

        return score


# Copied from transformers.models.deberta.modeling_deberta.DebertaAttention with Deberta->DebertaV2
class DebertaV2Attention(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config)
        self.config = config

    def construct(
        self,
        hidden_states,
        attention_mask,
        output_attentions: bool = False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        self_output, att_matrix = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)

        if output_attentions:
            return (attention_output, att_matrix)
        else:
            return (attention_output, None)


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->DebertaV2
class DebertaV2Intermediate(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaOutput with DebertaLayerNorm->LayerNorm
class DebertaV2Output(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaLayer with Deberta->DebertaV2
class DebertaV2Layer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.attention = DebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def construct(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions: bool = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        attention_output, att_matrix = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        if output_attentions:
            return (layer_output, att_matrix)
        else:
            return (layer_output, None)


class ConvLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        kernel_size = getattr(config, "conv_kernel_size", 3)
        groups = getattr(config, "conv_groups", 1)
        self.conv_act = getattr(config, "conv_act", "tanh")
        self.conv = mint.nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=groups,
        )
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.conv_act_fn = ACT2FN[self.conv_act]

    def construct(self, hidden_states, residual_states, input_mask):
        out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).broadcast_to(out.shape), 0)
        out = self.conv_act_fn(self.dropout(out))

        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input).to(layer_norm_input.dtype)

        if input_mask is None:
            output_states = output
        else:
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)

            input_mask = input_mask.to(output.dtype)
            output_states = output * input_mask

        return output_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaEmbeddings with DebertaLayerNorm->LayerNorm,Deberta->DebertaV2
class DebertaV2Embeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = mint.nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = mint.nn.Embedding(config.max_position_embeddings, self.embedding_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = mint.nn.Embedding(config.type_vocab_size, self.embedding_size)
        else:
            self.token_type_embeddings = None

        if self.embedding_size != config.hidden_size:
            self.embed_proj = mint.nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        else:
            self.embed_proj = None

        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = mint.arange(config.max_position_embeddings).broadcast_to((1, -1))

    def construct(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = mint.zeros(input_shape, dtype=ms.float32)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = mint.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings = embeddings + position_embeddings
        if self.token_type_embeddings is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        if self.embed_proj is not None:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


class DebertaV2Encoder(nn.Cell):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()

        self.layer = nn.CellList([DebertaV2Layer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            self.rel_embeddings = mint.nn.Embedding(pos_ebd_size, config.hidden_size)

        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        self.conv = ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        self.gradient_checkpointing = False

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            if query_states is not None:
                relative_pos = build_relative_position(
                    query_states,
                    hidden_states,
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                )
            else:
                relative_pos = build_relative_position(
                    hidden_states,
                    hidden_states,
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                )
        return relative_pos

    def construct(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = attention_mask.sum(-2) > 0
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states: Optional[Tuple[ms.Tensor]] = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        output_states = None

        next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        for i, layer_module in enumerate(self.layer):
            if self.gradient_checkpointing and self.training:
                raise NotImplementedError("Gradient checkpoint is not yet supported.")
            else:
                output_states, attn_weights = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

        if not return_dict:
            return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class DebertaV2PreTrainedModel(MSPreTrainedModel):
    config_class = DebertaV2Config
    base_model_prefix = "deberta"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.set_data(
                initializer(
                    Normal(sigma=self.config.initializer_range, mean=0.0), module.weight.shape, module.weight.dtype
                )
            )
            if module.bias is not None:
                module.bias.set_data(initializer(Zero(), module.bias.shape, module.bias.dtype))
        elif isinstance(module, mint.nn.Embedding):
            module.weight.set_data(
                initializer(
                    Normal(sigma=self.config.initializer_range, mean=0.0),
                    module.weight.shape,
                    module.weight.dtype,
                )
            )
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = 0
        elif isinstance(module, mint.nn.LayerNorm):
            module.weight.set_data(initializer(One(), module.weight.shape, module.weight.dtype))
            module.bias.set_data(initializer(Zero(), module.bias.shape, module.bias.dtype))
        elif isinstance(module, (LegacyDebertaV2LMPredictionHead, DebertaV2LMPredictionHead)):
            module.bias.set_data(initializer(Zero(), module.bias.shape, module.bias.dtype))


# Copied from transformers.models.deberta.modeling_deberta.DebertaModel with Deberta->DebertaV2
class DebertaV2Model(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)
        self.z_steps = 0
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError("The prune function is not implemented in DeBERTa model.")

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = mint.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = mint.zeros(input_shape, dtype=ms.int32)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )


# Copied from transformers.models.deberta.modeling_deberta.LegacyDebertaPredictionHeadTransform with Deberta->DebertaV2
class LegacyDebertaV2PredictionHeadTransform(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm((self.embedding_size,), epsilon=config.layer_norm_eps)

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LegacyDebertaV2LMPredictionHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.transform = LegacyDebertaV2PredictionHeadTransform(config)

        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = mint.nn.Linear(self.embedding_size, config.vocab_size, bias=False)

        self.bias = ms.Parameter(mint.zeros(config.vocab_size), name="bias")

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class LegacyDebertaV2OnlyMLMHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.predictions = LegacyDebertaV2LMPredictionHead(config)

    def construct(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class DebertaV2LMPredictionHead(nn.Cell):
    """https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/bert.py#L270"""

    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act

        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps, elementwise_affine=True)

        self.bias = ms.Parameter(mint.zeros(config.vocab_size), name="bias")

    # note that the input embeddings must be passed as an argument
    def construct(self, hidden_states, word_embeddings):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = mint.matmul(hidden_states, word_embeddings.weight.t()) + self.bias
        return hidden_states


class DebertaV2OnlyMLMHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.lm_head = DebertaV2LMPredictionHead(config)

    # note that the input embeddings must be passed as an argument
    def forward(self, sequence_output, word_embeddings):
        prediction_scores = self.lm_head(sequence_output, word_embeddings)
        return prediction_scores


class DebertaV2ForMaskedLM(DebertaV2PreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = r"mask_predictions.*"

    def __init__(self, config):
        super().__init__(config)
        self.legacy = config.legacy
        self.deberta = DebertaV2Model(config)
        if self.legacy:
            self.cls = LegacyDebertaV2OnlyMLMHead(config)
        else:
            self._tied_weights_keys = ["lm_predictions.lm_head.weight", "deberta.embeddings.word_embeddings.weight"]
            self.lm_predictions = DebertaV2OnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        if self.legacy:
            return self.cls.predictions.decoder
        else:
            return self.lm_predictions.lm_head.dense

    def set_output_embeddings(self, new_embeddings):
        if self.legacy:
            self.cls.predictions.decoder = new_embeddings
            self.cls.predictions.bias = new_embeddings.bias
        else:
            self.lm_predictions.lm_head.dense = new_embeddings
            self.lm_predictions.lm_head.bias = new_embeddings.bias

    # Copied from transformers.models.deberta.modeling_deberta.DebertaForMaskedLM.forward with Deberta->DebertaV2
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if self.legacy:
            prediction_scores = self.cls(sequence_output)
        else:
            prediction_scores = self.lm_predictions(sequence_output, self.deberta.embeddings.word_embeddings)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1).int())

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.deberta.modeling_deberta.ContextPooler
class ContextPooler(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mint.nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = mint.nn.Dropout(config.pooler_dropout)
        self.config = config
        self.pooler_act_fn = ACT2FN[config.pooler_hidden_act]

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = self.pooler_act_fn(pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


class DebertaV2ForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = mint.nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = mint.nn.Dropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.deberta.modeling_deberta.DebertaForSequenceClassification.forward with Deberta->DebertaV2
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.shape[-1] == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.shape[0] > 0:
                        labeled_logits = mint.gather(
                            logits, 0, label_index.broadcast_to((label_index.shape[0], logits.shape[1]))
                        )
                        labels = mint.gather(labels, 0, label_index.view(-1))
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1).int())
                    else:
                        loss = mint.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).int())
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


# Copied from transformers.models.deberta.modeling_deberta.DebertaForTokenClassification with Deberta->DebertaV2
class DebertaV2ForTokenClassification(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2Model(config)
        self.dropout = mint.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = mint.nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).int())

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class DebertaV2ForQuestionAnswering(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2Model(config)
        self.qa_outputs = mint.nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.deberta.modeling_deberta.DebertaForQuestionAnswering.forward with Deberta->DebertaV2
    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        start_positions: Optional[ms.Tensor] = None,
        end_positions: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions.int())
            end_loss = loss_fct(end_logits, end_positions.int())
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DebertaV2ForMultipleChoice(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = mint.nn.Linear(output_dim, 1)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = mint.nn.Dropout(drop_out)

        self.init_weights()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        token_type_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.shape[-1]) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        outputs = self.deberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.int())

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
