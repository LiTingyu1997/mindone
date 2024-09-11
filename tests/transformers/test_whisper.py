import pytest
import logging
from typing import Tuple, Union, Dict, List
import random
import numpy as np
import torch
from safetensors.torch import load_file
import mindspore as ms
from transformers import WhisperProcessor
from transformers.models.whisper import WhisperForConditionalGeneration as pt_whisper
from mindone.transformers.models.whisper import WhisperForConditionalGeneration as ms_whisper
from transformers.modeling_outputs import ModelOutput as pt_ModelOutput
from mindone.transformers.modeling_outputs import ModelOutput as ms_ModelOutput
from datasets import load_dataset

logger = logging.getLogger(__name__)


def test_ms_whisper(model_path):
    ms.set_context(mode=1, pynative_synchronize=True)
    # mindspore.set_context(mode=0, jit_syntax_level=mindspore.STRICT)

    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[0]["audio"]

    processor = WhisperProcessor.from_pretrained(model_path)
    model = ms_whisper.from_pretrained(model_path)
    model.config.forced_decoder_ids = None

    # audio file is decoded on the fly
    inputs = processor(sample["array"], sampling_rate=sampling_rate, return_tensors="np")
    for k, v in inputs.items():
        inputs[k] = ms.Tensor(v)

    predicted_ids = model.generate(input_features)

    # transcribe speech
    transcription = processor.batch_decode(predicted_ids)
    print(transcription[0])

@pytest.mark.parametrize(
    "name,mode,dtype",
    [
        ["openai/whisper-large", ms.GRAPH_MODE, "fp32"],
        ["openai/whisper-large", ms.GRAPH_MODE, "fp16"],
        ["openai/whisper-large", ms.PYNATIVE_MODE, "fp32"],
        ["openai/whisper-large", ms.PYNATIVE_MODE, "fp16"],
    ],
)
def test_whisper(model_name, mode, dtype):
    if mode == ms.GRAPH_MODE:
        ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    elif mode == ms.PYNATIVE_MODE:
        ms.set_context(mode=mode, pynative_synchronize=True)

    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[0]["audio"]

    processor = WhisperProcessor.from_pretrained(model_name)
    ms_model = ms_whisper.from_pretrained(model_name)
    pt_model = pt_whisper.from_pretrained(model_name)
    ms_model = ms_HubertForCTC.from_pretrained(model_name)

    # prepare inputs
    batch_data = _get_batch_data(dataset, batch_size=1, processor=processor, sampling_rate=sampling_rate)

    pt_inputs = {key: torch.tensor(value) for key, value in batch_data.items()}
    ms_inputs = {key: ms.tensor(value) for key, value in batch_data.items()}

    with torch.no_grad():
        pt_outputs = pt_model(**pt_inputs)
    ms_outputs = ms_model(**ms_inputs)

    diffs = _compute_diffs(pt_outputs.hidden_states, ms_outputs[1])
    pt_outputs = [x for x in pt_outputs.hidden_states]
    ms_outputs = [x.asnumpy() for x in ms_outputs[1]]
    pt_mean = np.mean(np.concatenate(pt_outputs))
    ms_mean = np.mean(np.concatenate(ms_outputs))

    print(f"pt_output_mean: {pt_mean}")
    print(f"ms_output_mean: {ms_mean}")
    print(f"relative error with respect to pytorch: {np.abs(pt_mean - ms_mean) / np.abs(pt_mean)}")
    print(f"diffs mean: {np.mean(diffs)}")


def _get_batch_data(dataset, batch_size, processor, sampling_rate):
    random_indices = random.sample(range(len(dataset)), batch_size)
    audio_arrays = [dataset[i]["audio"]["array"] for i in random_indices]
    inputs = processor(audio_arrays, sampling_rate=sampling_rate, return_tensors="np", padding=True)
    return inputs


def _compute_diffs(pt_outputs: torch.Tensor, ms_outputs: ms.Tensor):
    if isinstance(pt_outputs, pt_ModelOutput):
        pt_outputs = tuple(pt_outputs.values())
    elif not isinstance(pt_outputs, (tuple, list)):
        pt_outputs = (pt_outputs,)
    if not isinstance(ms_outputs, (tuple, list)):
        ms_outputs = (ms_outputs,)

    diffs = []
    for p, m in zip(pt_outputs, ms_outputs):
        if isinstance(p, pt_ModelOutput):
            p = tuple(p.values())[0]

        p = p.detach().cpu().numpy()
        m = m.asnumpy()

        # relative error defined by Frobenius norm
        # dist(x, y) := ||x - y|| / ||y||, where ||·|| means Frobenius norm
        d = np.linalg.norm(p - m) / np.linalg.norm(p)

        diffs.append(d)

    return diffs


if __name__ == "__main__":
    model_path = "/home/pingqi/.cache/huggingface/hub/models--facebook--hubert-large-ls960-ft/snapshots/ece5fabbf034c1073acae96d5401b25be96709d8"
    test_hubert(model_name=model_path, mode=0, dtype="fp32")