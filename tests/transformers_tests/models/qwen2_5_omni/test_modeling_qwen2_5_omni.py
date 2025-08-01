"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/qwen2_5_omni/test_modeling_qwen2_5_omni.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

# NOTE: need to install transformers by `pip install transformers>=4.52.0`

import inspect

import numpy as np
import pytest
import torch
from transformers.models.qwen2_5_omni import (  # Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniConfig,
    Qwen2_5OmniTalkerConfig,
    Qwen2_5OmniToken2WavConfig,
)
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (  # Qwen2_5OmniBigVGANConfig,; Qwen2_5OmniDiTConfig,
    Qwen2_5OmniTextConfig,
)

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-2, "fp16": 5e-4, "bf16": 5e-2}
MODES = [1]


class Qwen2_5_OmniModelTester:
    def __init__(
        self,
        batch_size=1,
        encoder_seq_length=7,
        decoder_seq_length=7,
        # For common tests
        is_training=False,
        use_attention_mask=True,
        use_labels=False,
        use_cache=False,
        output_attentions=False,
        # thinker
        vocab_size=99,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=32,
        max_position_embeddings=512,
        use_sliding_window=False,
        eos_token_id=1,
        attn_implementation="eager",
    ):
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.use_sliding_window = use_sliding_window
        self.attn_implementation = attn_implementation
        self.eos_token_id = eos_token_id
        self.rope_scaling = {
            "mrope_section": [2, 3, 3],
            "rope_type": "default",
            "type": "default",
        }  # sum*2=16 = head_dim = hidden_size//num_attention_heads = 128//8=16

    def get_large_model_config(self):
        return Qwen2_5OmniConfig.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    def prepare_config_and_inputs(self):
        input_ids = np.maximum(ids_numpy([self.batch_size, self.encoder_seq_length], self.vocab_size), 2)
        input_ids[:, -1] = self.eos_token_id  # Eos Token
        decoder_input_ids = ids_numpy([self.batch_size, self.decoder_seq_length], self.vocab_size)

        attention_mask = None
        decoder_attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_numpy([self.batch_size, self.encoder_seq_length], vocab_size=2)
            decoder_attention_mask = ids_numpy([self.batch_size, self.decoder_seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_numpy([self.batch_size, self.decoder_seq_length], self.vocab_size)

        thinker_config, talker_config, token2wav_config = self.get_config()

        return (
            thinker_config,
            talker_config,
            token2wav_config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    def get_config(self):
        thinker_config = Qwen2_5OmniTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            rope_scaling=self.rope_scaling,
            use_cache=self.use_cache,
            use_sliding_window=self.use_sliding_window,
            output_attentions=self.output_attentions,
            attn_implementation=self.attn_implementation,
        )

        talker_config = Qwen2_5OmniTalkerConfig(
            audio_token_index=151646,
            image_token_index=151655,
            video_token_index=151656,
            vocab_size=self.vocab_size,  # 8448,
            tts_codec_start_token_id=93,  # 8293,
            tts_codec_end_token_id=94,  # 8294,
            tts_codec_pad_token_id=92,  # 8292,
            tts_codec_mask_token_id=96,  # 8296,
            embedding_size=self.hidden_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            rope_scaling=self.rope_scaling,
            head_dim=16,
            use_cache=self.use_cache,
            use_sliding_window=self.use_sliding_window,  # False
            audio_start_token_id=151647,
            audio_end_token_id=151648,
            attn_implementation=self.attn_implementation,
        )

        # dit_config = Qwen2_5OmniDiTConfig()
        # bigvgan_config = Qwen2_5OmniBigVGANConfig()
        token2wav_config = Qwen2_5OmniToken2WavConfig()

        return thinker_config, talker_config, token2wav_config


model_tester = Qwen2_5_OmniModelTester()
(
    thinker_config,
    talker_config,
    token2wav_config,
    input_ids,
    decoder_input_ids,
    attention_mask,
    decoder_attention_mask,
    lm_labels,
) = model_tester.prepare_config_and_inputs()


T5_CASES = [
    [
        "Qwen2_5OmniThinkerTextModel",
        "transformers.models.qwen2_5_omni.Qwen2_5OmniThinkerTextModel",  # NOTE: name is different
        "mindone.transformers.Qwen2_5OmniThinkerTextModel",
        (thinker_config,),
        {},
        (),
        {"input_ids": input_ids, "attention_mask": attention_mask, "return_dict": True},
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
    [
        "Qwen2_5OmniTalkerModel",
        "transformers.Qwen2_5OmniTalkerModel",
        "mindone.transformers.Qwen2_5OmniTalkerModel",
        (talker_config,),
        {},
        (),
        {
            "input_ids": decoder_input_ids,
            "attention_mask": decoder_attention_mask,
        },
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
    # [
    #     "Qwen2_5OmniToken2WavModel",
    #     "transformers.Qwen2_5OmniToken2WavModel",
    #     "mindone.transformers.Qwen2_5OmniToken2WavModel",
    #     (token2wav_config,),
    #     {},
    #     (),
    #     {
    #         "code": code,
    #         "conditioning": conditioning,
    #         "reference_mel": reference_mel,
    #     },
    #     {},
    # ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [
        case
        + [
            dtype,
        ]
        + [
            mode,
        ]
        for case in T5_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODES
    ],
)
def test_named_modules(
    name,
    pt_module,
    ms_module,
    init_args,
    init_kwargs,
    inputs_args,
    inputs_kwargs,
    outputs_map,
    dtype,
    mode,
):
    ms.set_context(mode=mode)

    (
        pt_model,
        ms_model,
        pt_dtype,
        ms_dtype,
    ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    # set `hidden_dtype` if requiring, for some modules always compute in float
    # precision and require specific `hidden_dtype` to cast before return
    if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
        pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
        ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)
    # print("ms:", ms_outputs)
    # print("pt:", pt_outputs)
    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            # print("===map", pt_key, ms_idx)
            pt_output = getattr(pt_outputs, pt_key)
            # ms_output = ms_outputs[ms_idx]
            ms_output = getattr(ms_outputs, ms_idx)
            if isinstance(pt_output, (list, tuple)):
                pt_outputs_n += list(pt_output)
                ms_outputs_n += list(ms_output)
            else:
                pt_outputs_n.append(pt_output)
                ms_outputs_n.append(ms_output)
        diffs = compute_diffs(pt_outputs_n, ms_outputs_n)
    else:
        diffs = compute_diffs(pt_outputs, ms_outputs)

    THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < THRESHOLD).all(), (
        f"ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
    )
