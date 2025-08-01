<!-- Copyright 2024 The HuggingFace Team. All rights reserved.
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
# limitations under the License. -->

# LTX Video

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

[LTX Video](https://huggingface.co/Lightricks/LTX-Video) is the first DiT-based video generation model capable of generating high-quality videos in real-time. It produces 24 FPS videos at a 768x512 resolution faster than they can be watched. Trained on a large-scale dataset of diverse videos, the model generates high-resolution videos with realistic and varied content. We provide a model for both text-to-video as well as image + text-to-video usecases.

!!! tip

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

Available models:

|                                                             Model name                                                              | Recommended dtype |
|:-----------------------------------------------------------------------------------------------------------------------------------:|:-----------------:|
|             [`LTX Video 2B 0.9.0`](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.safetensors)             |   `ms.bfloat16`   |
|            [`LTX Video 2B 0.9.1`](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors)            |  `ms.bfloat16`    |
|            [`LTX Video 2B 0.9.5`](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.5.safetensors)            |   `ms.bfloat16`   |
|            [`LTX Video 13B 0.9.7`](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-dev.safetensors)            |   `ms.bfloat16`   |
| [`LTX Video Spatial Upscaler 0.9.7`](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-spatial-upscaler-0.9.7.safetensors) |   `ms.bfloat16`   |

Note: The recommended dtype is for the transformer component. The VAE and text encoders can be either `ms.float32`, `ms.bfloat16` or `ms.float16` but the recommended dtype is `ms.bfloat16` as used in the original repository.

## Recommended settings for generation

For the best results, it is recommended to follow the guidelines mentioned in the official LTX Video [repository](https://github.com/Lightricks/LTX-Video).

- Some variants of LTX Video are guidance-distilled. For guidance-distilled models, `guidance_scale` must be set to `1.0`. For any other models, `guidance_scale` should be set higher (e.g., `5.0`) for good generation quality.
- For variants with a timestep-aware VAE (LTXV 0.9.1 and above), it is recommended to set `decode_timestep` to `0.05` and `image_cond_noise_scale` to `0.025`.
- For variants that support interpolation between multiple conditioning images and videos (LTXV 0.9.5 and above), it is recommended to use similar looking images/videos for the best results. High divergence between the conditionings may lead to abrupt transitions in the generated video.

## Using LTX Video 13B 0.9.7

LTX Video 0.9.7 comes with a spatial latent upscaler and a 13B parameter transformer. The inference involves generating a low resolution video first, which is very fast, followed by upscaling and refining the generated video.

<!-- TODO(aryan): modify when official checkpoints are available -->

```python
import mindspore as ms
from mindone.diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from mindone.diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from mindone.diffusers.utils import export_to_video, load_video
import numpy as np

pipe = LTXConditionPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.7-diffusers", mindspore_dtype=ms.bfloat16)
pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.7-Latent-Spatial-Upsampler-diffusers", vae=pipe.vae, mindspore_dtype=ms.bfloat16)
pipe.vae.enable_tiling()

def round_to_nearest_resolution_acceptable_by_vae(height, width):
    height = height - (height % pipe.vae_temporal_compression_ratio)
    width = width - (width % pipe.vae_temporal_compression_ratio)
    return height, width

video = load_video(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
)[:21]  # Use only the first 21 frames as conditioning
condition1 = LTXVideoCondition(video=video, frame_index=0)

prompt = "The video depicts a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
expected_height, expected_width = 768, 1152
downscale_factor = 2 / 3
num_frames = 161

# Part 1. Generate video at smaller resolution
# Text-only conditioning is also supported without the need to pass `conditions`
downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
latents = pipe(
    conditions=[condition1],
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=downscaled_width,
    height=downscaled_height,
    num_frames=num_frames,
    num_inference_steps=30,
    generator=np.random.Generator(np.random.PCG64(seed=0)),
    output_type="latent",
)[0]

# Part 2. Upscale generated video using latent upsampler with fewer inference steps
# The available latent upsampler upscales the height/width by 2x
upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
upscaled_latents = pipe_upsample(
    latents=latents,
    output_type="latent"
)[0]

# Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
video = pipe(
    conditions=[condition1],
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=upscaled_width,
    height=upscaled_height,
    num_frames=num_frames,
    denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
    num_inference_steps=10,
    latents=upscaled_latents,
    decode_timestep=0.05,
    image_cond_noise_scale=0.025,
    generator=np.random.Generator(np.random.PCG64(seed=0)),
    output_type="pil",
)[0][0]

# Part 4. Downscale the video to the expected resolution
video = [frame.resize((expected_width, expected_height)) for frame in video]

export_to_video(video, "output.mp4", fps=24)
```

## Loading Single Files

Loading the original LTX Video checkpoints is also possible with [`~ModelMixin.from_single_file`]. We recommend using `from_single_file` for the Lightricks series of models, as they plan to release multiple models in the future in the single file format.

```python
import mindspore as ms
from mindone.diffusers import AutoencoderKLLTXVideo, LTXImageToVideoPipeline, LTXVideoTransformer3DModel

# `single_file_url` could also be https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.1.safetensors
single_file_url = "https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.safetensors"
transformer = LTXVideoTransformer3DModel.from_single_file(
  single_file_url, mindspore_dtype=ms.bfloat16
)
vae = AutoencoderKLLTXVideo.from_single_file(single_file_url, mindspore_dtype=ms.bfloat16)
pipe = LTXImageToVideoPipeline.from_pretrained(
  "Lightricks/LTX-Video", transformer=transformer, vae=vae, mindspore_dtype=ms.bfloat16
)

# ... inference code ...
```

Alternatively, the pipeline can be used to load the weights with [`~FromSingleFileMixin.from_single_file`].

```python
import mindspore as ms
from mindone.diffusers import LTXImageToVideoPipeline
from mindone.transformers import T5EncoderModel
from transformers import T5Tokenizer

single_file_url = "https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.safetensors"
text_encoder = T5EncoderModel.from_pretrained(
  "Lightricks/LTX-Video", subfolder="text_encoder", mindspore_dtype=ms.bfloat16
)
tokenizer = T5Tokenizer.from_pretrained(
  "Lightricks/LTX-Video", subfolder="tokenizer", mindspore_dtype=ms.bfloat16
)
pipe = LTXImageToVideoPipeline.from_single_file(
  single_file_url, text_encoder=text_encoder, tokenizer=tokenizer, mindspore_dtype=ms.bfloat16
)
```

Loading and running inference with [LTX Video 0.9.1](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors) weights.

```python
import mindspore as ms
from mindone.diffusers import LTXPipeline
from mindone.diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", mindspore_dtype=ms.bfloat16)

prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
)[0][0]
export_to_video(video, "output.mp4", fps=24)
```

Refer to [this section](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/cogvideox/#memory-optimization) to learn more about optimizing memory consumption.

::: mindone.diffusers.LTXPipeline

::: mindone.diffusers.LTXImageToVideoPipeline

::: mindone.diffusers.LTXConditionPipeline

::: mindone.diffusers.LTXLatentUpsamplePipeline

::: mindone.diffusers.pipelines.ltx.pipeline_output.LTXPipelineOutput
