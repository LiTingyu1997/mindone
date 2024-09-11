import pytest
import logging
from typing import Tuple, Union, Dict, List
import random
import numpy as np
import torch
from safetensors.torch import load_file
import mindspore as ms
from transformers import ClapProcessor
from transformers import ClapModel as pt_ClapModel
from mindone.transformers import ClapModel as ms_ClapModel
from transformers.modeling_outputs import ModelOutput as pt_ModelOutput
from datasets import load_dataset

logger = logging.getLogger(__name__)

THRESHOLD_FP16 = 1e-2
THRESHOLD_FP32 = 5e-3


def test_ms_clap(model_path):
    librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio_sample = librispeech_dummy[0]

    ms_model = ms_ClapModel.from_pretrained(model_path)
    processor = ClapProcessor.from_pretrained(model_path)

    inputs = processor(audios=audio_sample["audio"]["array"], return_tensors="np")
    ms_audio_embed = ms_model.get_audio_features(**inputs)
    print(ms_audio_embed.shape)

@pytest.mark.parametrize(
    "name,mode,dtype",
    [
        ["laion/clap-htsat-fused", ms.GRAPH_MODE, "fp32"],
        ["laion/clap-htsat-fused", ms.GRAPH_MODE, "fp16"],
        ["laion/clap-htsat-fused", ms.PYNATIVE_MODE, "fp32"],
        ["laion/clap-htsat-fused", ms.PYNATIVE_MODE, "fp16"],
    ],
)
def test_clap(model_name, mode, dtype):
    pass


def _get_batch_data(dataset, batch_size, processor, sampling_rate):
    random_indices = random.sample(range(len(dataset)), batch_size)
    audio_arrays = [dataset[i]["audio"]["array"] for i in random_indices]
    inputs = processor(audio_arrays, sampling_rate=sampling_rate, return_tensors="np", padding=True)
    return inputs


_TORCH_FP16_BLACKLIST = (
    "LayerNorm",
    "Timesteps",
    "AvgPool2d",
    "Upsample2D",
    "ResnetBlock2D",
    "FirUpsample2D",
    "FirDownsample2D",
    "KDownsample2D",
    "AutoencoderTiny",
)


def _set_model_dtype(pt_modules_instance, ms_modules_instance, dtype):
    if dtype == "fp16":
        pt_modules_instance = pt_modules_instance.to(torch.float16)
        ms_modules_instance = _set_dtype(ms_modules_instance, ms.float16)
    elif dtype == "fp32":
        pt_modules_instance = pt_modules_instance.to(torch.float32)
        ms_modules_instance = _set_dtype(ms_modules_instance, ms.float32)
    else:
        raise NotImplementedError(f"Dtype {dtype} for model is not implemented")

    pt_modules_instance.eval()
    ms_modules_instance.set_train(False)

    if dtype == "fp32":
        return pt_modules_instance, ms_modules_instance

    # Some torch modules do not support fp16 in CPU, converted to fp32 instead.
    for _, submodule in pt_modules_instance.named_modules():
        if submodule.__class__.__name__ in _TORCH_FP16_BLACKLIST:
            logger.warning(
                f"Model '{pt_modules_instance.__class__.__name__}' has submodule {submodule.__class__.__name__} which doens't support fp16, converted to fp32 instead."
            )
            pt_modules_instance = pt_modules_instance.to(torch.float32)
            break

    return pt_modules_instance, ms_modules_instance


def _set_dtype(model, dtype):
    for p in model.get_parameters():
        p = p.set_dtype(dtype)
    return model


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
    model_path = "/home/pingqi/.cache/huggingface/hub/models--laion--clap-htsat-fused/snapshots/cca9e288ab447cee67d9ada1f85ddb46500f1401"
    test_ms_clap(model_path)
    # test_hubert(model_name=model_path, mode=1, dtype="fp32")