# Copyright 2025 OmniGen team and The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from transformers import LlamaTokenizer

import mindspore as ms
from mindspore import mint

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...models.autoencoders import AutoencoderKL
from ...models.transformers import OmniGenTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.mindspore_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from .processor_omnigen import OmniGenMultiModalProcessor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import mindspore
        >>> from mindone.diffusers import OmniGenPipeline

        >>> pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1-diffusers", mindsporedtype=mindspore.bfloat16)

        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=50, guidance_scale=2.5)[0][0]
        >>> image.save("output.png")
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[ms.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and
        the second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class OmniGenPipeline(
    DiffusionPipeline,
):
    r"""
    The OmniGen pipeline for multimodal-to-image generation.

    Reference: https://huggingface.co/papers/2409.11340

    Args:
        transformer ([`OmniGenTransformer2DModel`]):
            Autoregressive Transformer architecture for OmniGen.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        tokenizer (`LlamaTokenizer`):
            Text tokenizer of class.
            [LlamaTokenizer](https://huggingface.co/docs/transformers/main/model_doc/llama#transformers.LlamaTokenizer).
    """

    model_cpu_offload_seq = "transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: OmniGenTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        tokenizer: LlamaTokenizer,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) is not None else 8
        )
        # OmniGen latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        self.multimodal_processor = OmniGenMultiModalProcessor(tokenizer, max_image_size=1024)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 120000
        )
        self.default_sample_size = 128

    def encode_input_images(
        self,
        input_pixel_values: List[ms.Tensor],
        dtype: Optional[ms.Type] = None,
    ):
        """
        get the continue embedding of input images by VAE

        Args:
            input_pixel_values: normalized pixel of input images
        Returns: ms.Tensor
        """
        dtype = dtype or self.vae.dtype

        input_img_latents = []
        for img in input_pixel_values:
            img = self.vae.diag_gauss_dist.sample(self.vae.encode(img.to(dtype))[0]).mul_(
                self.vae.config.scaling_factor
            )
            input_img_latents.append(img)
        return input_img_latents

    def check_inputs(
        self,
        prompt,
        input_images,
        height,
        width,
        use_input_image_size_as_output,
        callback_on_step_end_tensor_inputs=None,
    ):
        if input_images is not None:
            if len(input_images) != len(prompt):
                raise ValueError(
                    f"The number of prompts: {len(prompt)} does not match the number of input images: {len(input_images)}."
                )
            for i in range(len(input_images)):
                if input_images[i] is not None:
                    if not all(f"<img><|image_{k + 1}|></img>" in prompt[i] for k in range(len(input_images[i]))):
                        raise ValueError(
                            f"prompt `{prompt[i]}` doesn't have enough placeholders for the input images `{input_images[i]}`"
                        )

        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if use_input_image_size_as_output:
            if input_images is None or input_images[0] is None:
                raise ValueError(
                    "`use_input_image_size_as_output` is set to True, but no input image was found. \
                        If you are performing a text-to-image task, please set it to False."
                )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, \
                    but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, dtype=dtype)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def __call__(
        self,
        prompt: Union[str, List[str]],
        input_images: Union[PipelineImageInput, List[PipelineImageInput]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        max_input_image_size: int = 1024,
        timesteps: List[int] = None,
        guidance_scale: float = 2.5,
        img_guidance_scale: float = 1.6,
        use_input_image_size_as_output: bool = False,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        latents: Optional[ms.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If the input includes images, need to add
                placeholders `<img><|image_i|></img>` in the prompt to indicate the position of the i-th images.
            input_images (`PipelineImageInput` or `List[PipelineImageInput]`, *optional*):
                The list of input images. We will replace the "<|image_i|>" in prompt with the i-th image in list.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            max_input_image_size (`int`, *optional*, defaults to 1024):
                the maximum size of input image, which will be used to crop the input image to the maximum size
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 2.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            img_guidance_scale (`float`, *optional*, defaults to 1.6):
                Defined as equation 3 in [Instrucpix2pix](https://huggingface.co/papers/2211.09800).
            use_input_image_size_as_output (bool, defaults to False):
                whether to use the input image size as the output image size, which can be used for single-image input,
                e.g., image editing task
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`np.random.Generator` or `List[np.random.Generator]`, *optional*):
                One or a list of [`np.random.Generator`](https://numpy.org/doc/stable/reference/random/generator.html)
                to make generation deterministic.
            latents (`ms.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns: [`~pipelines.ImagePipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is returned
            where the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        num_cfg = 2 if input_images is not None else 1
        use_img_cfg = True if input_images is not None else False
        if isinstance(prompt, str):
            prompt = [prompt]
            input_images = [input_images]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            input_images,
            height,
            width,
            use_input_image_size_as_output,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Define call parameters
        batch_size = len(prompt)

        # 3. process multi-modal instructions
        if max_input_image_size != self.multimodal_processor.max_image_size:
            self.multimodal_processor.reset_max_image_size(max_image_size=max_input_image_size)
        processed_data = self.multimodal_processor(
            prompt,
            input_images,
            height=height,
            width=width,
            use_img_cfg=use_img_cfg,
            use_input_image_size_as_output=use_input_image_size_as_output,
            num_images_per_prompt=num_images_per_prompt,
        )

        # 4. Encode input images
        input_img_latents = self.encode_input_images(processed_data["input_pixel_values"])

        # 5. Prepare timesteps
        sigmas = np.linspace(1, 0, num_inference_steps + 1)[:num_inference_steps]
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, timesteps, sigmas=sigmas
        )
        self._num_timesteps = len(timesteps)

        # 6. Prepare latents
        transformer_dtype = self.transformer.dtype
        if use_input_image_size_as_output:
            height, width = processed_data["input_pixel_values"][0].shape[-2:]
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            ms.float32,
            generator,
            latents,
        )

        # 8. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = mint.cat([latents] * (num_cfg + 1))
                latent_model_input = latent_model_input.to(transformer_dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.broadcast_to((latent_model_input.shape[0],))

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    input_ids=processed_data["input_ids"],
                    input_img_latents=input_img_latents,
                    input_image_sizes=processed_data["input_image_sizes"],
                    attention_mask=processed_data["attention_mask"],
                    position_ids=processed_data["position_ids"],
                    return_dict=False,
                )[0]

                if num_cfg == 2:
                    cond, uncond, img_cond = mint.split(noise_pred, len(noise_pred) // 3, dim=0)
                    noise_pred = uncond + img_guidance_scale * (img_cond - uncond) + guidance_scale * (cond - img_cond)
                else:
                    cond, uncond = mint.split(noise_pred, len(noise_pred) // 2, dim=0)
                    noise_pred = uncond + guidance_scale * (cond - uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
