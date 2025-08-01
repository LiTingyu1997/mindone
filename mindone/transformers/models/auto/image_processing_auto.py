# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""AutoImageProcessor class."""

import importlib
import json
import os
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

# Build the list of all image processors
from transformers import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from transformers.utils import CONFIG_NAME, cached_file

from ...image_processing_utils import BaseImageProcessor, ImageProcessingMixin
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...utils import IMAGE_PROCESSOR_NAME, is_vision_available, logging
from .auto_factory import _LazyAutoMapping
from .configuration_auto import CONFIG_MAPPING_NAMES, AutoConfig, replace_list_option_in_docstrings

logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    # This significantly improves completion suggestion performance when
    # the transformers package is used with Microsoft's Pylance language server.
    IMAGE_PROCESSOR_MAPPING_NAMES: OrderedDict[str, Tuple[Optional[str], Optional[str]]] = OrderedDict()
else:
    IMAGE_PROCESSOR_MAPPING_NAMES = OrderedDict(
        [
            ("beit", ("BeitImageProcessor",)),
            ("blip", ("BlipImageProcessor",)),
            ("blip-2", ("BlipImageProcessor",)),
            ("chameleon", ("ChameleonImageProcessor",)),
            ("clip", ("CLIPImageProcessor",)),
            ("dpt", ("DPTImageProcessor",)),
            ("llava_next", ("LlavaNextImageProcessor",)),
            ("llava_next_video", ("LlavaNextVideoImageProcessor",)),
            ("llava_onevision", ("LlavaOnevisionImageProcessor",)),
        ]
    )

for model_type, image_processors in IMAGE_PROCESSOR_MAPPING_NAMES.items():
    slow_image_processor_class, *fast_image_processor_class = image_processors
    if not is_vision_available():
        slow_image_processor_class = None

    # If the fast image processor is not defined, or torchvision is not available, we set it to None
    if not fast_image_processor_class or fast_image_processor_class[0] is None:
        fast_image_processor_class = None
    else:
        fast_image_processor_class = fast_image_processor_class[0]

    IMAGE_PROCESSOR_MAPPING_NAMES[model_type] = (slow_image_processor_class, fast_image_processor_class)

IMAGE_PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, IMAGE_PROCESSOR_MAPPING_NAMES)


def image_processor_class_from_name(class_name: str):
    if class_name == "BaseImageProcessorFast":
        return BaseImageProcessorFast

    for module_name, extractors in IMAGE_PROCESSOR_MAPPING_NAMES.items():
        if class_name in extractors:
            sub_path = os.path.abspath(os.path.dirname(__file__))
            sub_path = str(Path(sub_path).parent)
            sys.path.insert(0, sub_path)
            module = importlib.import_module("mindone.transformers")

            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for _, extractors in IMAGE_PROCESSOR_MAPPING._extra_content.items():
        for extractor in extractors:
            if getattr(extractor, "__name__", None) == class_name:
                return extractor

    # We did not find the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def get_image_processor_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs,
):
    """
    Loads the image processor configuration from a pretrained model image processor configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the image processor configuration from local files.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the image processor.

    Examples:

    ```python
    # Download configuration from huggingface.co and cache.
    image_processor_config = get_image_processor_config("google-bert/bert-base-uncased")
    # This model does not have a image processor config so the result will be an empty dict.
    image_processor_config = get_image_processor_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained image processor locally and you can reload its config
    from transformers import AutoTokenizer

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_processor.save_pretrained("image-processor-test")
    image_processor_config = get_image_processor_config("image-processor-test")
    ```"""
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    resolved_config_file = cached_file(
        pretrained_model_name_or_path,
        IMAGE_PROCESSOR_NAME,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )
    if resolved_config_file is None:
        logger.info(
            "Could not locate the image processor configuration file, will try to use the model config instead."
        )
        return {}

    with open(resolved_config_file, encoding="utf-8") as reader:
        return json.load(reader)


def _warning_fast_image_processor_available(fast_class):
    logger.warning(
        f"Fast image processor class {fast_class} is available for this model. "
        "Using slow image processor class. To use the fast image processor class set `use_fast=True`."
    )


class AutoImageProcessor:
    r"""
    This is a generic image processor class that will be instantiated as one of the image processor classes of the
    library when created with the [`AutoImageProcessor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoImageProcessor is designed to be instantiated "
            "using the `AutoImageProcessor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(IMAGE_PROCESSOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        r"""
        Instantiate one of the image processor classes of the library from a pretrained model vocabulary.

        The image processor class to instantiate is selected based on the `model_type` property of the config object
        (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's
        missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained image_processor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a image processor file saved using the
                  [`~image_processing_utils.ImageProcessingMixin.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved image processor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model image processor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the image processor files and override the cached versions if
                they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            use_fast (`bool`, *optional*, defaults to `False`):
                Use a fast torchvision-base image processor if it is supported for a given model.
                If a fast tokenizer is not available for a given model, a normal numpy-based image processor
                is returned instead.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final image processor object. If `True`, then this
                functions returns a `Tuple(image_processor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not image processor attributes: i.e., the part of
                `kwargs` which has not been used to update `image_processor` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are image processor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* image processor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        <Tip>

        Passing `token=True` is required when you want to use a private model.

        </Tip>

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor

        >>> # Download image processor from huggingface.co and cache.
        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

        >>> # If image processor files are in a directory (e.g. image processor was saved using *save_pretrained('./test/saved_model/')*)
        >>> # image_processor = AutoImageProcessor.from_pretrained("./test/saved_model/")
        ```"""
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        config = kwargs.pop("config", None)
        use_fast = kwargs.pop("use_fast", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True

        config_dict, _ = ImageProcessingMixin.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)
        image_processor_class = config_dict.get("image_processor_type", None)
        image_processor_auto_map = None
        if "AutoImageProcessor" in config_dict.get("auto_map", {}):
            image_processor_auto_map = config_dict["auto_map"]["AutoImageProcessor"]

        # If we still don't have the image processor class, check if we're loading from a previous feature extractor config
        # and if so, infer the image processor class from there.
        if image_processor_class is None and image_processor_auto_map is None:
            feature_extractor_class = config_dict.pop("feature_extractor_type", None)
            if feature_extractor_class is not None:
                image_processor_class = feature_extractor_class.replace("FeatureExtractor", "ImageProcessor")
            if "AutoFeatureExtractor" in config_dict.get("auto_map", {}):
                feature_extractor_auto_map = config_dict["auto_map"]["AutoFeatureExtractor"]
                image_processor_auto_map = feature_extractor_auto_map.replace("FeatureExtractor", "ImageProcessor")

        # If we don't find the image processor class in the image processor config, let's try the model config.
        if image_processor_class is None and image_processor_auto_map is None:
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            # It could be in `config.image_processor_type``
            image_processor_class = getattr(config, "image_processor_type", None)
            if hasattr(config, "auto_map") and "AutoImageProcessor" in config.auto_map:
                image_processor_auto_map = config.auto_map["AutoImageProcessor"]

        if image_processor_class is not None:
            # Update class name to reflect the use_fast option. If class is not found, None is returned.
            if use_fast is not None:
                if use_fast and not image_processor_class.endswith("Fast"):
                    image_processor_class += "Fast"
                elif not use_fast and image_processor_class.endswith("Fast"):
                    image_processor_class = image_processor_class[:-4]
            image_processor_class = image_processor_class_from_name(image_processor_class)

        has_remote_code = image_processor_auto_map is not None
        has_local_code = image_processor_class is not None or type(config) in IMAGE_PROCESSOR_MAPPING
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        if image_processor_auto_map is not None and not isinstance(image_processor_auto_map, tuple):
            # In some configs, only the slow image processor class is stored
            image_processor_auto_map = (image_processor_auto_map, None)

        if has_remote_code and trust_remote_code:
            if not use_fast and image_processor_auto_map[1] is not None:
                _warning_fast_image_processor_available(image_processor_auto_map[1])

            if use_fast and image_processor_auto_map[1] is not None:
                class_ref = image_processor_auto_map[1]
            else:
                class_ref = image_processor_auto_map[0]
            image_processor_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
            _ = kwargs.pop("code_revision", None)
            if os.path.isdir(pretrained_model_name_or_path):
                image_processor_class.register_for_auto_class()
            return image_processor_class.from_dict(config_dict, **kwargs)
        elif image_processor_class is not None:
            return image_processor_class.from_dict(config_dict, **kwargs)
        # Last try: we use the IMAGE_PROCESSOR_MAPPING.
        elif type(config) in IMAGE_PROCESSOR_MAPPING:
            image_processor_tuple = IMAGE_PROCESSOR_MAPPING[type(config)]

            image_processor_class_py, image_processor_class_fast = image_processor_tuple

            if not use_fast and image_processor_class_fast is not None:
                _warning_fast_image_processor_available(image_processor_class_fast)

            if image_processor_class_fast and (use_fast or image_processor_class_py is None):
                return image_processor_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
            else:
                if image_processor_class_py is not None:
                    return image_processor_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
                else:
                    raise ValueError(
                        "This image processor cannot be instantiated. Please make sure you have `Pillow` installed."
                    )

        raise ValueError(
            f"Unrecognized image processor in {pretrained_model_name_or_path}. Should have a "
            f"`image_processor_type` key in its {IMAGE_PROCESSOR_NAME} of {CONFIG_NAME}, or one of the following "
            f"`model_type` keys in its {CONFIG_NAME}: {', '.join(c for c in IMAGE_PROCESSOR_MAPPING_NAMES.keys())}"
        )

    @staticmethod
    def register(
        config_class,
        image_processor_class=None,
        slow_image_processor_class=None,
        fast_image_processor_class=None,
        exist_ok=False,
    ):
        """
        Register a new image processor for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            image_processor_class ([`ImageProcessingMixin`]): The image processor to register.
        """
        if image_processor_class is not None:
            if slow_image_processor_class is not None:
                raise ValueError("Cannot specify both image_processor_class and slow_image_processor_class")
            warnings.warn(
                "The image_processor_class argument is deprecated and will be removed in v4.42. Please use `slow_image_processor_class`, "
                "or `fast_image_processor_class` instead",
                FutureWarning,
            )
            slow_image_processor_class = image_processor_class

        if slow_image_processor_class is None and fast_image_processor_class is None:
            raise ValueError("You need to specify either slow_image_processor_class or fast_image_processor_class")
        if slow_image_processor_class is not None and issubclass(slow_image_processor_class, BaseImageProcessorFast):
            raise ValueError("You passed a fast image processor in as the `slow_image_processor_class`.")
        if fast_image_processor_class is not None and issubclass(fast_image_processor_class, BaseImageProcessor):
            raise ValueError("You passed a slow image processor in as the `fast_image_processor_class`.")

        if (
            slow_image_processor_class is not None
            and fast_image_processor_class is not None
            and issubclass(fast_image_processor_class, BaseImageProcessorFast)
            and fast_image_processor_class.slow_image_processor_class != slow_image_processor_class
        ):
            raise ValueError(
                "The fast processor class you are passing has a `slow_image_processor_class` attribute that is not "
                "consistent with the slow processor class you passed (fast tokenizer has "
                f"{fast_image_processor_class.slow_image_processor_class} and you passed {slow_image_processor_class}. Fix one of those "
                "so they match!"
            )

        # Avoid resetting a set slow/fast image processor if we are passing just the other ones.
        if config_class in IMAGE_PROCESSOR_MAPPING._extra_content:
            existing_slow, existing_fast = IMAGE_PROCESSOR_MAPPING[config_class]
            if slow_image_processor_class is None:
                slow_image_processor_class = existing_slow
            if fast_image_processor_class is None:
                fast_image_processor_class = existing_fast

        IMAGE_PROCESSOR_MAPPING.register(
            config_class, (slow_image_processor_class, fast_image_processor_class), exist_ok=exist_ok
        )
