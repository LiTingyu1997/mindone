# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
import copy
import gc
import json
import os
import re
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import custom_object_save
from transformers.generation.configuration_utils import GenerationConfig
from transformers.safetensors_conversion import auto_conversion
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    DUMMY_INPUTS,
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ModelOutput,
    PushToHubMixin,
    cached_file,
    download_url,
    extract_commit_hash,
    find_adapter_config_file,
    has_file,
    is_offline_mode,
    is_remote_url,
    is_safetensors_available,
    logging,
    replace_return_docstrings,
)
from transformers.utils.hub import convert_file_size_to_int, get_checkpoint_shard_files

import mindspore as ms
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.nn import CrossEntropyLoss, Identity

from .activations import get_activation
from .generation.utils import GenerationMixin
from .integrations import PeftAdapterMixin
from .integrations.flash_attention import flash_attention_forward
from .integrations.sdpa_attention import sdpa_attention_forward
from .loss.loss_utils import LOSS_MAPPING
from .mindspore_adapter import dtype_to_str
from .mindspore_utils import (  # noqa: F401
    Conv1D,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
    prune_layer,
    prune_linear_layer,
)
from .modeling_attn_mask_utils import dtype_to_min
from .utils.import_utils import is_flash_attn_2_available, is_sdpa_available

if is_safetensors_available():
    from safetensors import safe_open

    # from mindone.safetensors.mindspore import load_file as safe_load_file
    from mindone.safetensors.mindspore import save_file as safe_save_file

logger = logging.get_logger(__name__)

_init_weights = True


def _get_pt2ms_mappings(m):
    mappings = {}  # pt_param_name: (ms_param_name, pt_param_to_ms_param_func)
    for name, cell in m.cells_and_names():
        if isinstance(cell, (nn.Conv1d, nn.Conv1dTranspose)):
            mappings[f"{name}.weight"] = f"{name}.weight", lambda x: ms.Parameter(
                ops.expand_dims(x, axis=-2), name=x.name
            )
            if "weight_norm_cell" in name:
                ori_name = name.replace(".weight_norm_cell", "")
                mappings[f"{ori_name}.weight_g"] = f"{ori_name}.weight_g", lambda x: ms.Parameter(
                    ops.expand_dims(x, axis=-2), name=x.name
                )
                mappings[f"{ori_name}.weight_v"] = f"{ori_name}.weight_v", lambda x: ms.Parameter(
                    ops.expand_dims(x, axis=-2), name=x.name
                )
                mappings[f"{ori_name}.bias"] = f"{name}.bias", lambda x: x
        elif isinstance(cell, nn.Embedding):
            mappings[f"{name}.weight"] = f"{name}.embedding_table", lambda x: x
        elif isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            mappings[f"{name}.weight"] = f"{name}.gamma", lambda x: x
            mappings[f"{name}.bias"] = f"{name}.beta", lambda x: x
            if isinstance(cell, (nn.BatchNorm2d,)):
                mappings[f"{name}.running_mean"] = f"{name}.moving_mean", lambda x: x
                mappings[f"{name}.running_var"] = f"{name}.moving_variance", lambda x: x
                mappings[f"{name}.num_batches_tracked"] = None, lambda x: x
    return mappings


def _get_pt2ms_mapped_k(mappings, has_prefix_module, expects_prefix_module, loaded_keys, prefix):
    if has_prefix_module and not expects_prefix_module:
        loaded_keys = [
            mappings.get(s[len(prefix) + 1 :], (s[len(prefix) + 1 :], lambda x: x))[0]
            if s.startswith(prefix)
            else mappings.get(s, (s, lambda x: x))[0]
            for s in loaded_keys
        ]
        loaded_keys = [".".join([prefix, s]) for s in loaded_keys]
    elif not has_prefix_module and expects_prefix_module:
        loaded_keys = [
            mappings.get(".".join([prefix, s]), (".".join([prefix, s]), lambda x: x))[0] for s in loaded_keys
        ]
        loaded_keys = [s[len(prefix) + 1 :] if s.startswith(prefix) else s for s in loaded_keys]
    else:
        loaded_keys = [mappings.get(s, (s, lambda x: x))[0] for s in loaded_keys]
    return loaded_keys


def _convert_state_dict(m, state_dict_pt, prefix=""):
    if not state_dict_pt:
        return state_dict_pt
    pt2ms_mappings = _get_pt2ms_mappings(m)
    state_dict_ms = {}
    while state_dict_pt:
        name_pt, data_pt = state_dict_pt.popitem()
        for name, param in m.parameters_and_names():
            name_ms = param.name
            length = len(prefix) + 1
            if name_pt.startswith(prefix):
                if name_ms.rsplit(".", 1)[0] == name_pt.rsplit(".", 1)[0][length:] or name_ms == name_pt[length:]:
                    name_pt = name_pt[length:]
            elif not name_pt.startswith(prefix):
                if name_pt.rsplit(".", 1)[0] == name_ms.rsplit(".", 1)[0][length:] or name_pt == name_ms[length:]:
                    name_pt = ".".join([prefix, name_pt])
        name_ms, data_mapping = pt2ms_mappings.get(name_pt, (name_pt, lambda x: x))
        data_ms = data_mapping(data_pt)
        if name_ms is not None:
            state_dict_ms[name_ms] = data_ms
    return state_dict_ms


@contextmanager
def silence_mindspore_logger():
    ms_logger = ms.log._get_logger()
    ms_level = ms_logger.level
    ms_logger.setLevel("ERROR")
    yield
    ms_logger.setLevel(ms_level)


def get_first_parameter_dtype(parameter: Union[nn.Cell, "ModuleUtilsMixin"]):
    """
    Returns the first parameter dtype (can be non-floating) or asserts if none were found.
    """
    return next(parameter.parameters_dict()).dtype


def get_parameter_dtype(parameter: Union[nn.Cell, "ModuleUtilsMixin"]):
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    last_dtype = None
    for t in parameter.get_parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            return t.dtype

    # if no floating dtype was found return whatever the first dtype is
    return last_dtype


def get_state_dict_dtype(state_dict):
    """
    Returns the first found floating dtype in `state_dict` if there is one, otherwise returns the first dtype.
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype

    # if no floating dtype was found return whatever the first dtype is
    return next(state_dict.values()).dtype


def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one ms.Parameter of type `dtype`.

    Example:

    ```py
    >>> dtype_byte_size(ms.float32)
    4
    ```
    """
    if dtype == ms.bool_:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def shard_checkpoint(
    state_dict: Dict[str, Tensor], max_shard_size: Union[int, str] = "10GB", weights_name: str = WEIGHTS_NAME
):
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size.

    The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so there is no
    optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For example, if the
    limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB],
    [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's weight is bigger than `max_shard_size`, it will end up in its own sub-checkpoint which will
    have a size greater than `max_shard_size`.

    </Tip>

    Args:
        state_dict (`Dict[str, Tensor]`): The state dictionary of a model to save.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
        weights_name (`str`, *optional*, defaults to `"pytorch_model.bin"`):
            The name of the model save file.
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = [{}]
    last_block_size = 0
    total_size = 0

    for key, weight in state_dict.items():
        weight_size = weight.numel() * dtype_byte_size(weight.dtype)

        # If this weight is going to tip up over the maximal size, we split, but only if we have put at least one
        # weight in the current shard.
        if last_block_size + weight_size > max_shard_size and len(sharded_state_dicts[-1]) > 0:
            sharded_state_dicts.append({})
            last_block_size = 0

        sharded_state_dicts[-1][key] = weight
        last_block_size += weight_size
        total_size += weight_size

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin")
        shard_file = shard_file.replace(".safetensors", f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors")
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index


def load_state_dict(checkpoint_file: Union[str, os.PathLike]):
    """
    Reads a PyTorch checkpoint file, returning properly formatted errors if they arise.
    """
    try:
        if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
            # Check format of the archive
            with safe_open(checkpoint_file, framework="np") as f:
                metadata = f.metadata()
            if metadata is not None and metadata.get("format") not in ["pt", "tf", "flax", "np"]:
                raise OSError(
                    f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                    "you save your model with the `save_pretrained` method."
                )
            return ms.load_checkpoint(checkpoint_file, format="safetensors")
        else:
            raise NotImplementedError(
                f"Only supports deserialization of weights file in safetensors format, but got {checkpoint_file}"
            )
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read(7) == "version":
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' " f"at '{checkpoint_file}'. "
            )


def _load_state_dict_into_model(model_to_load, state_dict, start_prefix, is_sharded=False):
    # # add prefix to the name of parameters
    # if len(start_prefix) > 0:
    #     for name, param in model_to_load.parameters_and_names():
    #         if param.name != name:
    #             logger.error(
    #                 f"When Loading state dict into model {model_to_load.__class__.__name__}, the attribute 'name' of 'mindspore.ms.Parameter' object is {param.name} which should be {name}.\n"  # noqa: E501
    #                 f"There are several possible reasons for this misalignment:\n"
    #                 f"  1. {model_to_load.__class__.__name__} didn't call 'MSPreTrainedModel.post_init()' correctly.\n"
    #                 f"  2. You have made changes to the model before loading the weights, which may be implicit. For example, you created an optimizer using the parameters of model.\n"  # noqa: E501
    #                 f"If you encounter this error, please report it to the developer."
    #             )
    #         param.name = start_prefix + name

    # TODO: error_msgs is always empty for now. Maybe we need to rewrite MindSpore's `load_param_into_net`.
    #  Error msgs should contain caught exception like size mismatch instead of missing/unexpected keys.
    # TODO: We should support loading float16 state_dict into float32 model, like PyTorch's behavior.
    error_msgs = []
    # TODO: State dict loading in mindspore does not cast dtype correctly. We do it manually. It's might unsafe.
    local_state = {v.name: v for k, v in model_to_load.parameters_and_names()}
    for k, v in state_dict.items():
        if k in local_state:
            v.set_dtype(local_state[k].dtype)
        else:
            pass  # unexpect key keeps origin dtype
    cm = silence_mindspore_logger() if is_sharded else nullcontext()
    with cm:
        ms.load_param_into_net(model_to_load, state_dict, strict_load=True)

    # remove prefix from the name of parameters
    if len(start_prefix) > 0:
        for name, param in model_to_load.parameters_and_names():
            param.name = name

    return error_msgs


def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


class ModuleUtilsMixin:
    """
    A few utilities for `mindspore.nn.Cell`, to be used as a mixin.
    """

    def _get_name(self):
        return self.__class__.__name__

    def to(self, dtype: Optional[ms.Type] = None):
        for p in self.get_parameters():
            p.set_dtype(dtype)
        return self

    def float(self):
        for p in self.get_parameters():
            p.set_dtype(ms.float32)
        return self

    def half(self):
        for p in self.get_parameters():
            p.set_dtype(ms.float16)
        return self

    @property
    def dtype(self) -> ms.Type:
        """
        `ms.Type`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`Tensor`): An attention mask.

        Returns:
            `Tensor`: The inverted attention mask.
        """
        encoder_extended_attention_mask = None
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        if encoder_extended_attention_mask is not None:
            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * dtype_to_min(self.dtype)

        return encoder_extended_attention_mask

    @staticmethod
    def create_extended_attention_mask_for_decoder(input_shape, attention_mask):
        batch_size, seq_length = input_shape
        seq_ids = ops.arange(seq_length)
        causal_mask = seq_ids[None, None, :].tile((batch_size, seq_length, 1)) <= seq_ids[None, :, None]
        causal_mask = causal_mask.to(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = ops.cat(
                [
                    ops.ones((batch_size, seq_length, prefix_seq_len), dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        # extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        extended_attention_mask = ops.mul(causal_mask.unsqueeze(1), attention_mask.unsqueeze(1).unsqueeze(1))
        return extended_attention_mask

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], dtype: ms.float32 = None
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `Tensor` The extended attention mask, with the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * dtype_to_min(dtype)
        return extended_attention_mask

    def get_head_mask(
        self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.tile((num_hidden_layers, 1, 1, 1, 1))
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.cells_and_names()
                if isinstance(module_type, nn.Embedding)
            ]
            total_parameters = [
                ms.Parameter for name, ms.Parameter in self.parameters_and_names() if name not in embedding_param_names
            ]
        else:
            total_parameters = list(self.get_parameters())

        total_numel = []
        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                total_numel.append(param.numel())

        return sum(total_numel)

    def estimate_tokens(self, input_dict: Dict[str, Union[ms.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (`dict`): The model inputs.

        Returns:
            `int`: The total number of tokens.
        """
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        if self.main_input_name in input_dict:
            return input_dict[self.main_input_name].numel()
        elif "estimate_tokens" not in self.warnings_issued:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            self.warnings_issued["estimate_tokens"] = True
        return 0

    def floating_point_ops(self, input_dict: Dict[str, Union[ms.Tensor, Any]], exclude_embeddings: bool = True) -> int:
        """
        Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
        batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
        tokens (valid if `12 * d_model << sequence_length`) as laid out in [this
        paper](https://arxiv.org/pdf/2001.08361.pdf) section 2.1. Should be overridden for transformers with parameter
        re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

        Args:
            batch_size (`int`):
                The batch size for the forward pass.

            sequence_length (`int`):
                The number of tokens in each line of the batch.

            exclude_embeddings (`bool`, *optional*, defaults to `True`):
                Whether or not to count embedding and softmax operations.

        Returns:
            `int`: The number of floating-point operations.
        """

        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)


class PreTrainedModel(nn.Cell, ModuleUtilsMixin, GenerationMixin, PushToHubMixin, PeftAdapterMixin):
    r"""
    Base class for all models.

    [`PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:

        - resize the input embeddings,
        - prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **load_tf_weights** (`Callable`) -- A python *method* for loading a TensorFlow checkpoint in a MindSpore model,
          taking as arguments:

            - **model** ([`PreTrainedModel`]) -- An instance of the model on which to load the TensorFlow checkpoint.
            - **config** ([`PreTrainedConfig`]) -- An instance of the configuration associated to the model.
            - **path** (`str`) -- A path to the TensorFlow checkpoint.

        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **is_parallelizable** (`bool`) -- A flag indicating whether this model supports model parallelization.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
    """

    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    model_tags = None

    _auto_class = None
    _no_split_modules = None
    _skip_keys_device_placement = None
    _keep_in_fp32_modules = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    # a list of `state_dict` keys that are potentially tied to another key in the state_dict.
    _tied_weights_keys = None

    is_parallelizable = False
    supports_gradient_checkpointing = False

    # Flash Attention 2 support
    _supports_flash_attn_2 = False

    # SDPA support
    _supports_sdpa = False

    # Has support for a `Cache` instance as `past_key_values`? Does it support a `StaticCache`?
    _supports_cache_class = False
    _supports_static_cache = False

    # Has support for dynamic model input?
    _supports_dynamic_input = False

    # Has support for a `QuantoQuantizedCache` instance as `past_key_values`
    _supports_quantized_cache = False

    # This flag signal that the model can be used as an efficient backend in TGI and vLLM
    # In practice, it means that they support attention interface functions, fully pass the kwargs
    # through all modules up to the Attention layer, can slice logits with Tensor, and have a default TP plan
    _supports_attention_backend = False

    @property
    def dummy_inputs(self) -> Dict[str, Tensor]:
        """
        `Dict[str, Tensor]`: Dummy inputs to do a forward pass in the network.
        """
        return {"input_ids": Tensor(DUMMY_INPUTS)}

    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a MindSpore model.
        """
        return "ms"

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"ms.Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # Save config and origin of the pretrained weights if given in model
        self.config = config
        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        # Overwrite the class attribute to make it an instance attribute, so models like
        # `InstructBlipForConditionalGeneration` can dynamically update it without modifying the class attribute
        # when a different component (e.g. language_model) is used.
        self._keep_in_fp32_modules = copy.copy(self.__class__._keep_in_fp32_modules)

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()

    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        use_flash_attention_2: bool = False,
        mindspore_dtype=None,
    ):
        """
        Automatically checks and dispatches to a default attention implementation. In order of priority:
            1. An implementation specified in `config._attn_implementation` (due for example to the argument attn_implementation="sdpa" in from_pretrained).
            2. DEPRECATED: if use_flash_attention_2 is set to `True` and `flash_attn` is available, flash attention. (`LlamaFlashAttention` for example)
            3. SDPA implementation, if available and supported by the model type. (`LlamaSdpaAttention` for example)
            4. The default model's implementation otherwise (`LlamaAttention` for example) .
        """
        # Here we use config._attn_implementation_internal to check whether the attention implementation was explicitely set by the user.
        # The property `PretrainedConfig._attn_implementation` is never `None`, for backward compatibility (always fall back on "eager").
        # The `hasattr` here is used as some Transformers tests for some reason do not call PretrainedConfig __init__ (e.g. test_no_super_init_config_and_model)
        requested_attn_implementation = None
        if hasattr(config, "_attn_implementation_internal") and config._attn_implementation_internal is not None:
            if config._attn_implementation != "flash_attention_2" and use_flash_attention_2:
                raise ValueError(
                    f'Both attn_implementation="{config._attn_implementation}" and `use_flash_attention_2=True` were '
                    f"used when loading the model, which are not compatible."
                    ' We recommend to just use `attn_implementation="flash_attention_2"` when loading the model.'
                )

            if config._attn_implementation not in ["eager", "paged_attention"] + ALL_ATTENTION_FUNCTIONS.valid_keys():
                message = (
                    f'Specified `attn_implementation="{config._attn_implementation}"` is not supported. '
                    f'The only possible arguments are `attn_implementation="eager"`'
                    f" (manual attention implementation)"
                )
                if cls._supports_flash_attn_2:
                    message += ', `"attn_implementation=flash_attention_2"` (implementation using flash attention 2)'
                if cls._supports_sdpa:
                    message += ', `"attn_implementation=sdpa"` (implementation using scaled_dot_product_attention)'
                raise ValueError(message + ".")

            # If a config is passed with a preset attn_implementation, we skip the automatic dispatch and use the
            # user-provided config, with hard checks that the requested attention implementation is available.
            requested_attn_implementation = config._attn_implementation_internal

        if use_flash_attention_2:
            logger.warning_once(
                "The model was loaded with use_flash_attention_2=True, which is deprecated and may be removed in a "
                'future release. Please use `attn_implementation="flash_attention_2"` instead.'
            )
            config._attn_implementation = "flash_attention_2"
        if config._attn_implementation == "flash_attention_2":
            cls._check_and_enable_flash_attn_2(
                config,
                mindspore_dtype=mindspore_dtype,
                hard_check_only=False,
            )
        elif requested_attn_implementation in [None, "sdpa"]:
            # use_flash_attention_2 takes priority over SDPA, hence SDPA treated in this elif.
            config = cls._check_and_enable_sdpa(
                config,
                hard_check_only=False if requested_attn_implementation is None else True,
            )

        return config

    @property
    def base_model(self) -> nn.Cell:
        """
        `mindspore.nn.Cell`: The main body of the model.
        """
        return getattr(self, self.base_model_prefix, self)

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()` from the `GenerationMixin`.

        Under the hood, on classes where this function returns True, some generation-specific changes are triggered:
        for instance, the model instance will have a populated `generation_config` attribute.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Directly inherits `GenerationMixin` -> can generate
        if "GenerationMixin" in str(cls.__bases__):
            return True
        # The class inherits from a class that can generate (recursive check) -> can generate
        for base in cls.__bases__:
            if not hasattr(base, "can_generate"):
                continue
            if "PreTrainedModel" not in str(base) and base.can_generate():
                return True
        # BC: Detects whether `prepare_inputs_for_generation` has been overwritten in the model. Prior to v4.45, this
        # was how we detected whether a model could generate.
        if "GenerationMixin" not in str(cls.prepare_inputs_for_generation):
            logger.warning_once(
                f"{cls.__name__} has generative capabilities, as `prepare_inputs_for_generation` is explicitly "
                "overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, "
                "`PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability "
                "to call `generate` and other related functions."
                "\n  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the "
                "model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes"
                "\n  - If you are the owner of the model architecture code, please modify your model class such that "
                "it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception)."
                "\n  - If you are not the owner of the model architecture class, please contact the model code owner "
                "to update it."
            )
            return True
        # Otherwise, can't generate
        return False

    @classmethod
    def _check_and_enable_flash_attn_2(
        cls,
        config,
        mindspore_dtype=None,
        hard_check_only: bool = False,
    ) -> PretrainedConfig:
        """
        Checks the availability of Flash Attention 2 and compatibility with the current model.

        If all checks pass and `hard_check_only` is False, the method will set the config attribute
        `attn_implementation` to "flash_attention_2" so that the model can initialize the correct attention module.
        """
        if not cls._supports_flash_attn_2:
            raise ValueError(
                f"{cls.__name__} does not support Flash Attention 2.0 yet. Please request to add support where"
                f" the model is hosted, on its model hub page: https://huggingface.co/{config._name_or_path}/discussions/new"
                " or in the Transformers GitHub repo: https://github.com/huggingface/transformers/issues/new"
            )

        if not is_flash_attn_2_available():
            raise ImportError("FlashAttention2 has been toggled on, but it cannot be used due to some error")

        if mindspore_dtype is None:
            logger.warning_once(
                "You are attempting to use Flash Attention 2.0 without specifying a MindSpore dtype. This might lead to unexpected behaviour"
            )
        elif mindspore_dtype is not None and mindspore_dtype not in [ms.float16, ms.bfloat16]:
            logger.warning_once(
                "Flash Attention 2.0 only supports ms.float16 and ms.bfloat16 dtypes, but"
                f" the current dype in {cls.__name__} is {mindspore_dtype}. You should run training or inference using "
                f"Automatic Mixed-Precision via the `network=auto_mix_precision(network, ...)` decorator,"
                " or load the model with the `mindspore_dtype` argument. Example: `model = "
                'AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", mindspore_dtype=ms.float16)`'
            )

        if not hard_check_only:
            config._attn_implementation = "flash_attention_2"
        return config

    @property
    def loss_function(self):
        if hasattr(self, "_loss_function"):
            return self._loss_function

        loss_type = getattr(self, "loss_type", None)

        if loss_type is None or loss_type not in LOSS_MAPPING:
            logger.warning_once(
                f"`loss_type={loss_type}` was set in the config but it is unrecognised."
                f"Using the default loss: `ForCausalLMLoss`."
            )
            loss_type = "ForCausalLM"
        return LOSS_MAPPING[loss_type]

    @loss_function.setter
    def loss_function(self, value):
        self._loss_function = value

    @classmethod
    def is_backend_compatible(cls):
        return cls._supports_attention_backend

    @classmethod
    def _check_and_enable_sdpa(cls, config, hard_check_only: bool = False) -> PretrainedConfig:
        """
        Checks the availability of SDPA for a given model.

        If all checks pass and `hard_check_only` is False, the method will set the config attribute `_attn_implementation`
        to "flash_attention_2" so that the model can initialize the correct attention module.
        """
        if hard_check_only:
            if not cls._supports_sdpa:
                raise ValueError(
                    f"{cls.__name__} does not support an attention implementation through `scaled_dot_product_attention` yet."
                    " Please request the support for this architecture: https://github.com/huggingface/transformers/issues/28005. "
                    "If you believe this error is a bug, please open an issue in Transformers GitHub repository and "
                    'load your model with the argument `attn_implementation="eager"` meanwhile. Example: '
                    '`model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="eager")`'
                )
            if not is_sdpa_available():
                raise ImportError("SDPA requirements in Transformers are not met.")

        if not is_sdpa_available() or not cls._supports_sdpa:
            return config

        if not hard_check_only:
            config._attn_implementation = "sdpa"
        return config

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            torch_dtype (str, *optional*):
                Override the default torch_dtype and load the model under this dtype.
        """
        # when we init a model from within another model (e.g. VLMs) and dispatch on FA2
        # a warning is raised that dtype should be fp16. Since we never pass dtype from within
        # modeling code, we can try to infer it here same way as done in `from_pretrained`
        if hasattr(config, "mindspore_dtype"):
            mindspore_dtype = kwargs.pop("mindspore_dtype", config.mindspore_dtype)
        else:
            mindspore_dtype = kwargs.pop("torch_dtype", config.torch_dtype)

        if isinstance(mindspore_dtype, str):
            mindspore_dtype = getattr(ms, mindspore_dtype)
        elif mindspore_dtype is not None:
            TORCH_TO_MINDSPORE_DTYPE_MAP = {
                "torch.float32": ms.float32,
                "torch.bfloat16": ms.bfloat16,
                "torch.float16": ms.float16,
            }
            mindspore_dtype = str(mindspore_dtype)
            mindspore_dtype = TORCH_TO_MINDSPORE_DTYPE_MAP[mindspore_dtype]

        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in _from_config.

        if config._attn_implementation_internal is not None:
            # In this case, the config has been created with the attn_implementation set by the user, which we
            # should respect.
            attn_implementation = config._attn_implementation_internal
        else:
            attn_implementation = None

        config._attn_implementation = kwargs.pop("attn_implementation", attn_implementation)
        if not getattr(config, "_attn_implementation_autoset", False):
            config = cls._autoset_attn_implementation(
                config,
                use_flash_attention_2=use_flash_attention_2,
                mindspore_dtype=mindspore_dtype,
            )

        model = cls(config, **kwargs)

        # We cannot set default mindspore dtype. So we need to cast model weights after creating.
        if mindspore_dtype is not None:
            model = model.to(mindspore_dtype)

            logger.info(
                f"convert model:{model.__class__.__name__} parameters to mindspore_dtype {dtype_to_str(mindspore_dtype)}"
            )

        return model

    def get_input_embeddings(self) -> nn.Cell:
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Cell`: A mindspore cell mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def set_input_embeddings(self, value: nn.Cell):
        """
        Set model's input embeddings.

        Args:
            value (`nn.Cell`): A cell mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def get_output_embeddings(self) -> nn.Cell:
        """
        Returns the model's output embeddings.

        Returns:
            `nn.Cell`: A mindspore cell mapping hidden states to vocabulary.
        """
        return None  # Overwrite for models with output embeddings

    def _init_weights(self, module):
        """
        Initialize the weights. This method should be overridden by derived class and is
        the only initialization method that will be called when loading a checkpoint
        using `from_pretrained`. Any attempt to initialize outside of this function
        will be useless as the mindspore.common.initializer function are all replaced with skip.
        """
        pass

    def _initialize_weights(self, module):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_hf_initialized", False):
            return
        self._init_weights(module)
        module._is_hf_initialized = True

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        if getattr(self.config.get_text_config(decoder=True), "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            tied_weights = self._tie_encoder_decoder_weights(
                self.encoder, self.decoder, self.base_model_prefix, "encoder"
            )
            # Setting a dynamic variable instead of `_tied_weights_keys` because it's a class
            # attributed not an instance member, therefore modifying it will modify the entire class
            # Leading to issues on subsequent calls by different tests or subsequent calls.
            self._dynamic_tied_weights_keys = tied_weights

        for name, module in self.cells_and_names():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    @staticmethod
    def _tie_encoder_decoder_weights(
        encoder: nn.Cell, decoder: nn.Cell, base_model_prefix: str, base_encoder_name: str
    ):
        uninitialized_encoder_weights: List[str] = []
        tied_weights: List[str] = []
        if decoder.__class__ != encoder.__class__:
            logger.info(
                f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder"
                " weights are correctly initialized."
            )

        def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Cell,
            encoder_pointer: nn.Cell,
            module_name: str,
            base_encoder_name: str,
            uninitialized_encoder_weights: List[str],
            depth=0,
            total_decoder_name="",
            total_encoder_name="",
        ):
            assert isinstance(decoder_pointer, nn.Cell) and isinstance(
                encoder_pointer, nn.Cell
            ), f"{decoder_pointer} and {encoder_pointer} have to be of type nn.Module"
            if hasattr(decoder_pointer, "weight"):
                assert hasattr(encoder_pointer, "weight")
                encoder_pointer.weight = decoder_pointer.weight
                tied_weights.append(f"{base_encoder_name}{total_encoder_name}.weight")
                if hasattr(decoder_pointer, "bias"):
                    assert hasattr(encoder_pointer, "bias")
                    tied_weights.append(f"{base_encoder_name}{total_encoder_name}.bias")
                    encoder_pointer.bias = decoder_pointer.bias
                return

            encoder_modules = encoder_pointer._modules
            decoder_modules = decoder_pointer._modules
            if len(decoder_modules) > 0:
                assert (
                    len(encoder_modules) > 0
                ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

                all_encoder_weights = {module_name + "/" + sub_name for sub_name in encoder_modules.keys()}
                encoder_layer_pos = 0
                for name, module in decoder_modules.items():
                    if name.isdigit():
                        encoder_name = str(int(name) + encoder_layer_pos)
                        decoder_name = name
                        if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                            encoder_modules
                        ) != len(decoder_modules):
                            # this can happen if the name corresponds to the position in a list module list of layers
                            # in this case the decoder has added a cross-attention that the encoder does not have
                            # thus skip this step and subtract one layer pos from encoder
                            encoder_layer_pos -= 1
                            continue
                    elif name not in encoder_modules:
                        continue
                    elif depth > 500:
                        raise ValueError(
                            "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is"
                            " a circular dependency between two or more `nn.Modules` of your model."
                        )
                    else:
                        decoder_name = encoder_name = name
                    tie_encoder_to_decoder_recursively(
                        decoder_modules[decoder_name],
                        encoder_modules[encoder_name],
                        module_name + "/" + name,
                        base_encoder_name,
                        uninitialized_encoder_weights,
                        depth=depth + 1,
                        total_encoder_name=f"{total_encoder_name}.{encoder_name}",
                        total_decoder_name=f"{total_decoder_name}.{decoder_name}",
                    )
                    all_encoder_weights.remove(module_name + "/" + encoder_name)

                uninitialized_encoder_weights += list(all_encoder_weights)

        # tie weights recursively
        tie_encoder_to_decoder_recursively(
            decoder, encoder, base_model_prefix, base_encoder_name, uninitialized_encoder_weights
        )

        if len(uninitialized_encoder_weights) > 0:
            logger.warning(
                f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}"
            )
        return tied_weights

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        if self.config.torchscript:
            try:
                output_embeddings.weight = Parameter(input_embeddings.embedding_table)
            except AttributeError:
                # in case of mint.nn.Embedding
                output_embeddings.weight = Parameter(input_embeddings.weight)
        else:
            try:
                output_embeddings.weight = input_embeddings.embedding_table
            except AttributeError:
                # in case of mint.nn.Embedding
                output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias = mint.nn.functional.pad(
                output_embeddings.bias,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `mindspore.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

        Return:
            `mindspore.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = model_embeds.embedding_table.shape[0]
        self.vocab_size = model_embeds.embedding_table.shape[0]

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)

        old_embeddings_requires_grad = old_embeddings.embedding_table.requires_grad
        new_embeddings.embedding_table.requires_grad = old_embeddings_requires_grad
        self.set_input_embeddings(new_embeddings)

        # Update new_num_tokens with the actual size of new_embeddings
        if pad_to_multiple_of is not None:
            new_num_tokens = new_embeddings.embedding_table.shape[0]

        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head = self.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            old_lm_head_requires_grad = old_lm_head.weight.requires_grad
            new_lm_head.weight.requires_grad = old_lm_head_requires_grad
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`mindspore.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `mindspore.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

        Return:
            `mindspore.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            `new_num_tokens` is `None`
        """

        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int):
                raise ValueError(
                    f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, "
                    f"which is not and integer. Please make sure to pass an integer"
                )
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.embedding_table.shape[0]
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        else:
            logger.info(
                "You are resizing the embedding layer without providing a `pad_to_multiple_of` ms.Parameter. This means that the new embedding"
                f" dimension will be {new_num_tokens}. This might induce some performance reduction as *Tensor Cores* will not be available."
                " For more details about this, or help on choosing the correct value for resizing, refer to this guide:"
                " https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc"
            )

        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.embedding_table.shape

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
        )
        new_embeddings.embedding_table.set_dtype(old_embeddings.embedding_table.dtype)
        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.embedding_table.data[:n, :] = old_embeddings.embedding_table.data[:n, :]

        # Replace weights in old_embeddings and return to maintain the same embedding type.
        # This ensures correct functionality when a Custom Embedding class is passed as input.
        # The input and output embedding types remain consistent. (c.f. https://github.com/huggingface/transformers/pull/31979)
        old_embeddings.embedding_table.set_data(new_embeddings.embedding_table.data)
        old_embeddings.num_embeddings = new_embeddings.embedding_table.data.shape[0]
        if old_embeddings.padding_idx is not None and (new_num_tokens - 1) < old_embeddings.padding_idx:
            old_embeddings.padding_idx = None

        return new_embeddings

    def _get_resized_lm_head(
        self, old_lm_head: nn.Dense, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
    ) -> nn.Dense:
        """
        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_lm_head (`mindspore.nn.Dense`):
                Old lm head liner layer to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `mindspore.nn.Dense` module of the model without doing anything. transposed (`bool`, *optional*, defaults
                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
                vocab_size` else `vocab_size, lm_head_dim`.

        Return:
            `mindspore.nn.Dense`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
            `None`
        """
        if new_num_tokens is None:
            return old_lm_head

        old_num_tokens, old_lm_head_dim = (
            old_lm_head.weight.shape if not transposed else old_lm_head.weight.transpose().shape
        )

        if old_num_tokens == new_num_tokens:
            return old_lm_head

        if not isinstance(old_lm_head, nn.Dense):
            raise TypeError(
                f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Dense}. You"
                " should either use a different resize function or make sure that `old_lm_head` are an instance of"
                f" {nn.Dense}."
            )

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None

        new_lm_head = nn.Dense(
            *new_lm_head_shape,
            has_bias=has_new_lm_head_bias,
            dtype=old_lm_head.weight.dtype,
        )

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        self._copy_lm_head_original_to_resized(
            new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
        )

        return new_lm_head

    def _init_added_embeddings_weights_with_mean(
        self, old_embeddings, new_embeddings, old_embedding_dim, old_num_tokens, added_num_tokens
    ):
        old_embeddings_weight = old_embeddings.weight.to(ms.float32)
        mean_embeddings = mint.mean(old_embeddings_weight, axis=0)

        # Check if the covariance is positive definite.
        is_covariance_psd = False
        if is_covariance_psd:
            raise NotImplementedError
        else:
            # Otherwise, just initialize with the mean. because distribution will not be created.
            new_embeddings.weight[-1 * added_num_tokens :, :] = (
                mean_embeddings[None, :].repeat(added_num_tokens, 1).to(old_embeddings.weight.dtype)
            )

    def _init_added_lm_head_weights_with_mean(
        self,
        old_lm_head,
        new_lm_head,
        old_lm_head_dim,
        old_num_tokens,
        added_num_tokens,
        transposed=False,
    ):
        if transposed:
            # Transpose to the desired shape for the function.
            new_lm_head.weight = new_lm_head.weight.t()
            old_lm_head.weight.data = old_lm_head.weight.t()

        # The same initialization logic as Embeddings.
        self._init_added_embeddings_weights_with_mean(
            old_lm_head, new_lm_head, old_lm_head_dim, old_num_tokens, added_num_tokens
        )

        if transposed:
            # Transpose again to the correct shape.
            new_lm_head.weight = new_lm_head.weight.t()
            old_lm_head.weight = old_lm_head.weight.t()

    def _init_added_lm_head_bias_with_mean(self, old_lm_head, new_lm_head, added_num_tokens):
        bias_mean = mint.mean(old_lm_head.bias.data, axis=0, dtype=ms.float32)
        bias_std = mint.std(old_lm_head.bias.data, axis=0).to(ms.float32)
        new_lm_head.bias.data[-1 * added_num_tokens :].normal_(mean=bias_mean, std=1e-9 * bias_std)

    def _copy_lm_head_original_to_resized(
        self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
    ):
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        raise NotImplementedError(
            f"`get_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        if _init_weights:
            # Initialize weights
            self.apply(self._initialize_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways
            self.tie_weights()

        # MindSpore patch. Refresh name of parameters.
        for name, param in self.parameters_and_names():
            param.name = name

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = ms.save_checkpoint,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~PreTrainedModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            state_dict (nested dictionary of `Tensor`):
                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `ms.save_checkpoint` by another method.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                We default it to 5GB in order for models to be able to run easily on free-tier google colab instances
                without CPU OOM issues.

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            save_peft_format (`bool`, *optional*, defaults to `True`):
                For backward compatibility with PEFT library, in case adapter weights are attached to the model, all
                keys of the state dict of adapters needs to be pre-pended with `base_model.model`. Advanced users can
                disable this behaviours by setting `save_peft_format` to `False`.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        _hf_peft_config_loaded = getattr(self, "_hf_peft_config_loaded", False)

        if "save_config" in kwargs:
            warnings.warn(
                "`save_config` is deprecated and will be removed in v5 of Transformers. Use `is_main_process` instead."
            )
            is_main_process = kwargs.pop("save_config")
        if safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self  # we don't unwrap_model(self) in mindspore

        # save the string version of dtype to the config, e.g. convert ms.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = repr(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        # Save the config
        if is_main_process:
            if not _hf_peft_config_loaded:
                model_to_save.config.save_pretrained(save_directory)
            if self.can_generate():
                # generation config built from the model config + the model config holds generation kwargs -> generate
                # may revert to legacy behavior if the two don't match
                if (
                    model_to_save.generation_config._from_model_config
                    and model_to_save.config._get_non_default_generation_parameters()
                ):
                    new_generation_config = GenerationConfig.from_model_config(model_to_save.config)
                    if new_generation_config != model_to_save.generation_config:
                        logger.warning(
                            "Your generation config was originally created from the model config, but the model "
                            "config has changed since then. Unless you pass the `generation_config` argument to this "
                            "model's `generate` calls, they will revert to the legacy behavior where the base "
                            "`generate` parameterization is loaded from the model config instead. "
                            "To avoid this behavior and this warning, we recommend you to overwrite the generation "
                            "config model attribute before calling the model's `save_pretrained`, preferably also "
                            "removing any generation kwargs from the model config. This warning will be raised to an "
                            "exception in v4.41."
                        )
                model_to_save.generation_config.save_pretrained(save_directory)

            if _hf_peft_config_loaded:
                logger.info(
                    "Detected adapters on the model, saving the model in the PEFT format, only adapter weights will be saved."
                )
                state_dict = model_to_save.get_adapter_state_dict()

                if save_peft_format:
                    logger.info(
                        "To match the expected format of the PEFT library, all keys of the state dict of adapters will "
                        "be pre-pended with `base_model.model`."
                    )
                    peft_state_dict = {}
                    for key, value in state_dict.items():
                        peft_state_dict[f"base_model.model.{key}"] = value
                    state_dict = peft_state_dict

                active_adapter = self.active_adapters()

                if len(active_adapter) > 1:
                    raise ValueError(
                        "Multiple active adapters detected, saving multiple active adapters is not supported yet. "
                        "You can save adapters separately one by one "
                        "by iteratively calling `model.set_adapter(adapter_name)` then `model.save_pretrained(...)`"
                    )
                active_adapter = active_adapter[0]

                current_peft_config = self.peft_config[active_adapter]
                current_peft_config.save_pretrained(save_directory)

        # Save the model
        if state_dict is None:
            state_dict = {k: v for k, v in model_to_save.parameters_and_names()}

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            for ignore_key in self._keys_to_ignore_on_save:
                if ignore_key in state_dict.keys():
                    del state_dict[ignore_key]

        # Shard the model if it is too big.
        if not _hf_peft_config_loaded:
            weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
            weights_name = _add_variant(weights_name, variant)
        else:
            weights_name = ADAPTER_SAFE_WEIGHTS_NAME if safe_serialization else ADAPTER_WEIGHTS_NAME

        shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")

            # make sure that file to be deleted matches format of sharded file, e.g. mindspore_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
                and is_main_process
                and reg.fullmatch(filename_no_suffix) is not None
            ):
                os.remove(full_filename)

        # Save the model
        for shard_file, shard in shards.items():
            if safe_serialization:
                # At some point we will need to deal better with save_function (used for TPU and other distributed
                # joyfulness), but for now this enough.
                safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "np"})
            else:
                save_function(shard, os.path.join(save_directory, shard_file))

        if index is None:
            path_to_weights = os.path.join(save_directory, weights_name)
            logger.info(f"Model weights saved in {path_to_weights}")
        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        r"""
        Instantiate a pretrained mindspore model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      MindSpore model using the provided conversion scripts and loading the MindSpore model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PretrainedConfig`],
                    - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (`Dict[str, Tensor]`, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (`bool`, *optional*, defaults to `False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            mindspore_dtype (`str` or `mindspore.Type`, *optional*):
                Override the default `mindspore.Type` and load the model under a specific `dtype`. The different options
                are:

                1. `ms.float16` or `ms.bfloat16` or `ms.float32`: load in a specified
                  `dtype`, ignoring the model's `config.mindspore_dtype` if one exists. If not specified
                  - the model will get loaded in `ms.float32` (fp32).

                2. `"auto"` - A `mindspore_dtype` entry in the `config.json` file of the model will be
                  attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
                  the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
                  using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
                  the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.

                <Tip>

                For some models the `dtype` they were trained in is unknown - you may try to check the model's paper or
                reach out to the authors and ask them to add this information to the model's card and to insert the
                `mindspore_dtype` entry in `config.json` on the hub.

                </Tip>

            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* mindspore_model.<variant>.bin. `variant` is
                ignored when using `from_tf` or `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`
                is not installed, it will be set to `False`.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        Examples:

        ```python
        >>> from transformers import BertConfig, BertModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = BertModel.from_pretrained("./test/saved_model/")
        >>> # Update configuration during loading.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
        >>> assert model.config.output_attentions == True
        >>> # Loading from a TF checkpoint file instead of a MindSpore model (slower, for example purposes, not runnable).
        >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
        >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
        >>> # Loading from a Flax checkpoint file instead of a MindSpore model (slower)
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", from_flax=True)
        ```

        * `low_cpu_mem_usage` algorithm:

        This is an experimental function that loads the model using ~1x model size CPU memory

        Here is how it works:

        1. save which state_dict keys we have
        2. drop state_dict before the model is created, since the latter takes 1x model size CPU memory
        3. after the model has been instantiated switch to the meta device all params/buffers that
        are going to be replaced from the loaded state_dict
        4. load state_dict 2nd time
        5. replace the params/buffers from the state_dict

        Currently, it can't handle deepspeed ZeRO stage 3 and ignores loading errors

        """
        state_dict = kwargs.pop("state_dict", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        _ = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        mindspore_dtype = kwargs.pop("mindspore_dtype", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        variant = kwargs.pop("variant", None)
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
        adapter_kwargs = kwargs.pop("adapter_kwargs", {})
        adapter_name = kwargs.pop("adapter_name", "default")

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None and adapter_kwargs is not None and "token" not in adapter_kwargs:
            adapter_kwargs["token"] = token

        if use_safetensors is None and not is_safetensors_available():
            use_safetensors = False

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        try:
            _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)
        except Exception:
            _adapter_model_path = None
            adapter_kwargs = {}

        if _adapter_model_path is None:
            _adapter_model_path = find_adapter_config_file(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                _commit_hash=commit_hash,
                **adapter_kwargs,
            )
        if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
            with open(_adapter_model_path, "r", encoding="utf-8") as f:
                _adapter_model_path = pretrained_model_name_or_path
                pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]

        from_pt = not (from_tf | from_flax)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            # In case one passes a config to `from_pretrained` + "attn_implementation"
            # override the `_attn_implementation` attribute to `attn_implementation` of the kwargs
            # Please see: https://github.com/huggingface/transformers/issues/28038

            # Overwrite `config._attn_implementation` by the one from the kwargs --> in auto-factory
            # we pop attn_implementation from the kwargs but this handles the case where users
            # passes manually the config to `from_pretrained`.
            config = copy.deepcopy(config)

            kwarg_attn_imp = kwargs.pop("attn_implementation", None)
            if kwarg_attn_imp is not None and config._attn_implementation != kwarg_attn_imp:
                config._attn_implementation = kwarg_attn_imp
            model_kwargs = kwargs

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        # Load model
        loading_info = None

        # Keep in fp32 modules
        keep_in_fp32_modules = None
        use_keep_in_fp32_modules = False

        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
                ):
                    # Load from a TF 1.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
                ):
                    # Load from a TF 2.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
                elif from_flax and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
                ):
                    # Load from a Flax checkpoint in priority if from_flax
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
                ):
                    # Load from a safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant)
                    )
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                    )
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
                    )
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
                ):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
                ) or os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)):
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path} but there is a file for TensorFlow weights. Use"
                        " `from_tf=True` to load this model from those weights."
                    )
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)):
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path} but there is a file for Flax weights. Use `from_flax=True`"
                        " to load this model from those weights."
                    )
                elif use_safetensors:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME},"
                        f" {TF_WEIGHTS_NAME + '.index'} or {FLAX_WEIGHTS_NAME} found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path + ".index")):
                if not from_tf:
                    raise ValueError(
                        f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                        "from_tf to True to load from this checkpoint."
                    )
                archive_file = os.path.join(subfolder, pretrained_model_name_or_path + ".index")
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                # set correct filename
                if from_tf:
                    filename = TF2_WEIGHTS_NAME
                elif from_flax:
                    filename = FLAX_WEIGHTS_NAME
                elif use_safetensors is not False:
                    filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
                else:
                    filename = _add_variant(WEIGHTS_NAME, variant)

                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = {
                        "cache_dir": cache_dir,
                        "force_download": force_download,
                        "proxies": proxies,
                        "resume_download": resume_download,
                        "local_files_only": local_files_only,
                        "token": token,
                        "user_agent": user_agent,
                        "revision": revision,
                        "subfolder": subfolder,
                        "_raise_exceptions_for_gated_repo": False,
                        "_raise_exceptions_for_missing_entries": False,
                        "_commit_hash": commit_hash,
                    }
                    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                        elif use_safetensors:
                            if revision == "main":
                                resolved_archive_file, revision, is_sharded = auto_conversion(
                                    pretrained_model_name_or_path, **cached_file_kwargs
                                )
                            cached_file_kwargs["revision"] = revision
                            if resolved_archive_file is None:
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                                    "and thus cannot be loaded with `safetensors`. Please make sure that the model has "
                                    "been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                                )
                        else:
                            # This repo has no safetensors file of any kind, we switch to PyTorch.
                            filename = _add_variant(WEIGHTS_NAME, variant)
                            resolved_archive_file = cached_file(
                                pretrained_model_name_or_path, filename, **cached_file_kwargs
                            )
                    if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None:
                        # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                        # message.
                        has_file_kwargs = {
                            "revision": revision,
                            "proxies": proxies,
                            "token": token,
                        }
                        if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for TensorFlow weights."
                                " Use `from_tf=True` to load this model from those weights."
                            )
                        elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for Flax weights. Use"
                                " `from_flax=True` to load this model from those weights."
                            )
                        elif variant is not None and has_file(
                            pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs
                        ):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                                f" {variant}. Use `variant=None` to load this model from those weights."
                            )
                        else:
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {_add_variant(WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or"
                                f" {FLAX_WEIGHTS_NAME}."
                            )
                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception as e:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)},"
                        f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                    ) from e

            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=commit_hash,
            )

        if (
            is_safetensors_available()
            and isinstance(resolved_archive_file, str)
            and resolved_archive_file.endswith(".safetensors")
        ):
            with safe_open(resolved_archive_file, framework="np") as f:
                metadata = f.metadata()

            if metadata.get("format") in ("np", "pt"):
                pass
            elif metadata.get("format") == "tf":
                from_tf = True
                logger.info("A TensorFlow safetensors file is being loaded in a MindSpore model.")
            elif metadata.get("format") == "flax":
                from_flax = True
                logger.info("A Flax safetensors file is being loaded in a PyTorch model.")
            else:
                raise ValueError(
                    f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax'] but {metadata.get('format')}"
                )

        from_pt = not (from_tf | from_flax)

        # load pt weights early so that we know which dtype to init the model under
        if from_pt:
            if not is_sharded and state_dict is None:
                # Time to load the checkpoint
                state_dict = load_state_dict(resolved_archive_file)

            # set dtype to instantiate the model under:
            # 1. If mindspore_dtype is not None, we use that dtype
            # 2. If mindspore_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
            #    weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
            # we also may have config.torch_dtype available, but we won't rely on it till v5

            if mindspore_dtype is not None:
                config.mindspore_dtype = dtype_to_str(mindspore_dtype)
                for sub_config_key in config.sub_configs.keys():
                    sub_config = getattr(config, sub_config_key)
                    sub_config.mindspore_dtype = mindspore_dtype
                if isinstance(mindspore_dtype, str):
                    if mindspore_dtype == "auto":
                        if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
                            mindspore_dtype = config.torch_dtype
                            logger.info(f"Will use dtype={mindspore_dtype} as defined in model's config object")
                        else:
                            if is_sharded and "dtype" in sharded_metadata:
                                mindspore_dtype = sharded_metadata["dtype"]
                            elif not is_sharded:
                                mindspore_dtype = get_state_dict_dtype(state_dict)
                            else:
                                one_state_dict = load_state_dict(resolved_archive_file[0])
                                mindspore_dtype = get_state_dict_dtype(one_state_dict)
                                del one_state_dict  # free CPU memory
                            logger.info(
                                f"Since the `torch_dtype` attribute can't be found in model's config object, "
                                f"will use dtype={mindspore_dtype} as derived from model's weights"
                            )
                    else:
                        raise ValueError(
                            f'`mindspore_dtype` can be either `ms.Type` or `"auto"`, but received {mindspore_dtype}'
                        )
                # TODO: We cannot set default mindspore dtype!

            # Check if `_keep_in_fp32_modules` is not None
            use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (mindspore_dtype == ms.float16)

            if is_sharded:
                loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
            else:
                loaded_state_dict_keys = list(state_dict.keys())

        config.name_or_path = pretrained_model_name_or_path

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        config = cls._autoset_attn_implementation(
            config, use_flash_attention_2=use_flash_attention_2, mindspore_dtype=mindspore_dtype
        )

        model = cls(config, *model_args, **model_kwargs)

        # Make sure to tie the weights correctly
        model.tie_weights()

        # We cannot set default mindspore dtype. So we need to cast model weights after creating.
        if mindspore_dtype is not None:
            model = model.to(mindspore_dtype)

            logger.info(
                f"convert model:{model.__class__.__name__} parameters to mindspore_dtype {dtype_to_str(mindspore_dtype)}"
            )

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        # Check first if we are `from_pt`
        if use_keep_in_fp32_modules:
            keep_in_fp32_modules = model._keep_in_fp32_modules
        else:
            keep_in_fp32_modules = []

        if from_tf:
            raise NotImplementedError("loading tf checkpoint in mindspore model is not yet supported.")
        elif from_flax:
            raise NotImplementedError("loading flax checkpoint in mindspore model is not yet supported.")
        elif from_pt:
            (
                model,
                missing_keys,
                unexpected_keys,
                mismatched_keys,
                error_msgs,
            ) = cls._load_pretrained_model(
                model,
                state_dict,
                loaded_state_dict_keys,  # XXX: rename?
                resolved_archive_file,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                sharded_metadata=sharded_metadata,
                dtype=mindspore_dtype,
                keep_in_fp32_modules=keep_in_fp32_modules,
            )

        if _adapter_model_path is not None:
            model.load_adapter(
                _adapter_model_path,
                adapter_name=adapter_name,
                token=token,
                adapter_kwargs=adapter_kwargs,
            )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.set_train(False)

        # If it is a model with generation capabilities, attempt to load the generation config
        if model.can_generate() and pretrained_model_name_or_path is not None:
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **kwargs,
                )
            except OSError:
                logger.info(
                    "Generation config file not found, using a generation config created from the model config."
                )
                pass

        if output_loading_info:
            if loading_info is None:
                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }
            return model, loading_info

        return model

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
        sharded_metadata=None,
        dtype=None,
        keep_in_fp32_modules=None,
    ):
        model.tie_weights()

        # Retrieve missing & unexpected_keys
        model_state_dict = {k: v for k, v in model.parameters_and_names()}
        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix
        original_loaded_keys = loaded_keys

        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        # Mapping loaded_keys from pt to ms
        pt2ms_mappings = _get_pt2ms_mappings(model)
        loaded_keys = _get_pt2ms_mapped_k(pt2ms_mappings, has_prefix_module, expects_prefix_module, loaded_keys, prefix)

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            _prefix = f"{prefix}."
            expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
            expected_keys = [s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        missing_keys = sorted(set(expected_keys) - set(loaded_keys))
        unexpected_keys = set(loaded_keys) - set(expected_keys)

        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        # Set some modules to fp32 if any
        if keep_in_fp32_modules is not None:
            for name, param in model.parameters_and_names():
                if any(module_to_keep_in_fp32 in name.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules):
                    param.set_dtype(ms.float32)

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if len(cls.base_model_prefix) > 0 and not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        if len(cls.base_model_prefix) > 0 and hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)
            base_model_expected_keys = list(k for k, v in model_to_load.parameters_and_names())
            if any(key in expected_keys_not_prefixed and key not in base_model_expected_keys for key in loaded_keys):
                raise ValueError(
                    "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                    "properly saved?"
                )

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    # If the checkpoint is sharded, we may not have the key here.
                    if checkpoint_key not in state_dict:
                        continue
                    model_key = checkpoint_key
                    if remove_prefix_from_model:
                        # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                        model_key = f"{prefix}.{checkpoint_key}"
                    elif add_prefix_to_model:
                        # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                        model_key = ".".join(checkpoint_key.split(".")[1:])

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        if (
                            state_dict[checkpoint_key].shape[-1] == 1
                            and state_dict[checkpoint_key].numel() * 2 == model_state_dict[model_key].numel()
                        ):
                            # This skips size mismatches for 4-bit weights. Two 4-bit values share an 8-bit container, causing size differences.
                            # Without matching with module type or paramter type it seems like a practical way to detect valid 4bit weights.
                            pass
                        else:
                            mismatched_keys.append(
                                (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                            )
                            del state_dict[checkpoint_key]
            return mismatched_keys

        if state_dict is not None:
            # Whole checkpoint
            state_dict = _convert_state_dict(model, state_dict, prefix)

            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )
            error_msgs = _load_state_dict_into_model(model_to_load, state_dict, start_prefix, is_sharded=False)
        else:
            # Sharded checkpoint or whole but low_cpu_mem_usage==True

            # This should always be a list but, just to be sure.
            if not isinstance(resolved_archive_file, list):
                resolved_archive_file = [resolved_archive_file]

            error_msgs = []
            mismatched_keys = []

            if len(resolved_archive_file) > 1:
                resolved_archive_file = logging.tqdm(resolved_archive_file, desc="Loading checkpoint shards")

            # loading checkpoint
            for shard_file in resolved_archive_file:
                state_dict = load_state_dict(shard_file)
                state_dict = _convert_state_dict(model, state_dict, prefix)

                # Mismatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
                # matching the weights in the model.
                mismatched_keys += _find_mismatched_keys(
                    state_dict,
                    model_state_dict,
                    original_loaded_keys,
                    add_prefix_to_model,
                    remove_prefix_from_model,
                    ignore_mismatched_sizes,
                )

                error_msgs += _load_state_dict_into_model(model_to_load, state_dict, start_prefix, is_sharded=True)

                # force memory release
                del state_dict
                gc.collect()

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            archs = [] if model.config.architectures is None else model.config.architectures
            warner = logger.warning if model.__class__.__name__ in archs else logger.info
            warner(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs

    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        module_keys = {".".join(key.split(".")[:-1]) for key in names}

        # torch.nn.ParameterList is a special case where two parameter keywords
        # are appended to the module name, *e.g.* bert.special_embeddings.0
        module_keys = module_keys.union(
            {".".join(key.split(".")[:-2]) for key in names if len(key) > 0 and key[-1].isdigit()}
        )

        retrieved_modules = []
        # retrieve all modules that has at least one missing weight name
        for name, module in self.named_modules():
            if remove_prefix:
                _prefix = f"{self.base_model_prefix}."
                name = name[len(_prefix) :] if name.startswith(_prefix) else name
            elif add_prefix:
                name = ".".join([self.base_model_prefix, name]) if len(name) > 0 else self.base_model_prefix

            if name in module_keys:
                retrieved_modules.append(module)

        return retrieved_modules

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoModel"`):
                The auto class to register this new model with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    def warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
        """
        Shows a one-time warning if the input_ids appear to contain padding and no attention mask was given.
        """

        if (attention_mask is not None) or (self.config.pad_token_id is None):
            return

        # Check only the first and last input IDs to reduce overhead.
        if self.config.pad_token_id in input_ids[:, [-1, 0]]:
            warn_string = (
                "We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See "
                "https://huggingface.co/docs/transformers/troubleshooting"
                "#incorrect-output-when-padding-tokens-arent-masked."
            )

            # If the pad token is equal to either BOS, EOS, or SEP, we do not know whether the user should use an
            # attention_mask or not. In this case, we should still show a warning because this is a rare case.
            if (
                (self.config.bos_token_id is not None and self.config.bos_token_id == self.config.pad_token_id)
                or (self.config.eos_token_id is not None and self.config.eos_token_id == self.config.pad_token_id)
                or (self.config.sep_token_id is not None and self.config.sep_token_id == self.config.pad_token_id)
            ):
                warn_string += (
                    f"\nYou may ignore this warning if your `pad_token_id` ({self.config.pad_token_id}) is identical "
                    f"to the `bos_token_id` ({self.config.bos_token_id}), `eos_token_id` ({self.config.eos_token_id}), "
                    f"or the `sep_token_id` ({self.config.sep_token_id}), and your input is not padded."
                )

            logger.warning_once(warn_string)


class PoolerStartLogits(nn.Cell):
    """
    Compute SQuAD start logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, 1)

    def construct(self, hidden_states: ms.Tensor, p_mask: Optional[ms.Tensor] = None) -> ms.Tensor:
        """
        Args:
            hidden_states (`ms.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            p_mask (`ms.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
            `ms.Tensor`: The start logits for SQuAD.
        """
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            if get_parameter_dtype(self) == ms.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerEndLogits(nn.Cell):
    """
    Compute SQuAD end logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense_0 = mint.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = mint.nn.Tanh()
        self.LayerNorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = mint.nn.Linear(config.hidden_size, 1)

    def construct(
        self,
        hidden_states: ms.Tensor,
        start_states: Optional[ms.Tensor] = None,
        start_positions: Optional[ms.Tensor] = None,
        p_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        """
        Args:
            hidden_states (`ms.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`ms.Tensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`ms.Tensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            p_mask (`ms.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        <Tip>

        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.

        </Tip>

        Returns:
            `ms.Tensor`: The end logits for SQuAD.
        """
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(mint.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if get_parameter_dtype(self) == ms.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerAnswerClass(nn.Cell):
    """
    Compute SQuAD 2.0 answer class from classification and start tokens hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_0 = mint.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = mint.nn.Tanh()
        self.dense_1 = mint.nn.Linear(config.hidden_size, 1, bias=False)

    def construct(
        self,
        hidden_states: ms.Tensor,
        start_states: Optional[ms.Tensor] = None,
        start_positions: Optional[ms.Tensor] = None,
        cls_index: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        """
        Args:
            hidden_states (`ms.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`ms.Tensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`ms.Tensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            cls_index (`ms.Tensor` of shape `(batch_size,)`, *optional*):
                Position of the CLS token for each sentence in the batch. If `None`, takes the last token.

        <Tip>

        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.

        </Tip>

        Returns:
            `ms.Tensor`: The SQuAD 2.0 answer class.
        """
        # No dependency on end_feature so that we can obtain one single `cls_logits` for each sample.
        hsz = hidden_states.shape[-1]
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)  # shape (bsz, hsz)

        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        x = self.dense_0(mint.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)

        return x


@dataclass
class SquadHeadOutput(ModelOutput):
    """
    Base class for outputs of question answering models using a [`~modeling_utils.SQuADHead`].

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        start_top_log_probs (`mindspore.Tensor` of shape `(batch_size, config.start_n_top)`, *optional*,
            returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (`mindspore.Tensor` of shape `(batch_size, config.start_n_top)`, *optional*,
            returned if `start_positions` or `end_positions` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (`mindspore.Tensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*,
            returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
            (beam-search).
        end_top_index (`mindspore.Tensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*,
            returned if `start_positions` or `end_positions` is not provided):
            Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
        cls_logits (`mindspore.Tensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the `is_impossible` label of the answers.

    """

    loss: Optional[ms.Tensor] = None
    start_top_log_probs: Optional[ms.Tensor] = None
    start_top_index: Optional[ms.Tensor] = None
    end_top_log_probs: Optional[ms.Tensor] = None
    end_top_index: Optional[ms.Tensor] = None
    cls_logits: Optional[ms.Tensor] = None


class SQuADHead(nn.Cell):
    r"""
    A SQuAD head inspired by XLNet.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config):
        super().__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    @replace_return_docstrings(output_type=SquadHeadOutput, config_class=PretrainedConfig)
    def construct(
        self,
        hidden_states: ms.Tensor,
        start_positions: Optional[ms.Tensor] = None,
        end_positions: Optional[ms.Tensor] = None,
        cls_index: Optional[ms.Tensor] = None,
        is_impossible: Optional[ms.Tensor] = None,
        p_mask: Optional[ms.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[SquadHeadOutput, Tuple[ms.Tensor]]:
        """
        Args:
            hidden_states (`mindspore.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                Final hidden states of the model on the sequence tokens.
            start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Positions of the first token for the labeled span.
            end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Positions of the last token for the labeled span.
            cls_index (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Position of the CLS token for each sentence in the batch. If `None`, takes the last token.
            is_impossible (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Whether the question has a possible answer in the paragraph or not.
            p_mask (`mindspore.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
        """
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5

            return SquadHeadOutput(loss=total_loss) if return_dict else (total_loss,)

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = mint.softmax(start_logits, dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = mint.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = mint.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = mint.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = mint.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = mint.einsum("blh,bl->bh", hidden_states, start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)

            if not return_dict:
                return (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)
            else:
                return SquadHeadOutput(
                    start_top_log_probs=start_top_log_probs,
                    start_top_index=start_top_index,
                    end_top_log_probs=end_top_log_probs,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )


class SequenceSummary(nn.Cell):
    r"""
    Compute a single vector summary of a sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_type** (`str`) -- The method to use to make this summary. Accepted values are:

                - `"last"` -- Take the last token hidden state (like XLNet)
                - `"first"` -- Take the first token hidden state (like Bert)
                - `"mean"` -- Take the mean of all tokens hidden states
                - `"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
                - `"attn"` -- Not implemented now, use multi-head attention

            - **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
              (otherwise to `config.hidden_size`).
            - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
              another string or `None` will add no activation.
            - **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
            - **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.summary_type = getattr(config, "summary_type", "last")
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Dense(config.hidden_size, num_classes)

        activation_string = getattr(config, "summary_activation", None)
        self.activation: Callable = get_activation(activation_string) if activation_string else Identity()

        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def construct(self, hidden_states: ms.Tensor, cls_index: Optional[ms.Tensor] = None) -> ms.Tensor:
        """
        Compute a single vector summary of a sequence hidden states.

        Args:
            hidden_states (`torch.FloatTensor` of shape `[batch_size, seq_len, hidden_size]`):
                The hidden states of the last layer.
            cls_index (`torch.LongTensor` of shape `[batch_size]` or `[batch_size, ...]`
            where ... are optional leading dimensions of `hidden_states`, *optional*):
                Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.

        Returns:
            `torch.FloatTensor`: The summary of the sequence hidden states.
        """
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = hidden_states.mean(axis=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = ops.full_like(
                    hidden_states[..., :1, :],
                    hidden_states.shape[-2] - 1,
                    dtype=ms.int64,
                )
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


class AttentionInterface(MutableMapping):
    """
    Dict-like object keeping track of allowed attention functions. You can easily add a new attention function
    with a call to `register()`. If a model needs to locally overwrite an existing attention function, say `sdpa`,
    it needs to declare a new instance of this class inside the `modeling_<model>.py`, and declare it on that instance.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {
        "flash_attention_2": flash_attention_forward,
        # "flex_attention": flex_attention_forward,  # Mindspore dose not support flex_attention yet
        "sdpa": sdpa_attention_forward,  # Mindspore dose not support sdpa yet. Use vanilla attention to work around
    }

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key):
        # First check if instance has a local override
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key, value):
        # Allow local update of the default functions without impacting other instances
        self._local_mapping.update({key: value})

    def __delitem__(self, key):
        del self._local_mapping[key]

    def __iter__(self):
        # Ensure we use all keys, with the overwritten ones on top
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self):
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    @classmethod
    def register(cls, key: str, value: Callable):
        cls._global_mapping.update({key: value})

    def valid_keys(self) -> List[str]:
        return list(self.keys())


# Global AttentionInterface shared by all models which do not need to overwrite any of the existing ones
ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()

# for BC
MSPreTrainedModel = PreTrainedModel
