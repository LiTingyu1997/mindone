# Stable Diffusion text-to-image fine-tuning

The `train_text_to_image.py` script shows how to fine-tune stable diffusion model on your own dataset.

___Note___:

___This script is experimental. The script fine-tunes the whole model and often times the model overfits and runs into issues like catastrophic forgetting. It's recommended to try different hyperparameters to get the best result on your dataset.___


## Running locally with MindSpore

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/mindspore-lab/mindone
cd mindone
pip install -e ".[training]"
```


### OnePiece example

You need to accept the model license before downloading or using the weights. In this example we'll use model version `v1-4`, so you'll need to visit [its card](https://huggingface.co/CompVis/stable-diffusion-v1-4), read the license and tick the checkbox if you agree.

You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section of the documentation](https://huggingface.co/docs/hub/security-tokens).

Run the following command to authenticate your token

```bash
huggingface-cli login
```

If you have already cloned the repo, then you won't need to go through these steps.

<br>

#### Hardware

With `gradient_checkpointing` and `mixed_precision` it should be possible to fine tune the model on a single 24GB NPU. For higher `batch_size` and faster training it's better to use NPUs with >30GB memory.

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="YaYaB/onepiece-blip-captions"

python train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --mixed_precision="fp16" \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-onepiece-model-$(date +%Y%m%d%H%M%S)"
```

To run on your own training files prepare the dataset according to the format required by `datasets`, you can find the instructions for how to do that in this [document](https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder-with-metadata).
If you wish to use custom loading logic, you should modify the script, we have left pointers for that in the training script.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="path_to_your_dataset"

python train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --mixed_precision="fp16" \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-your-dataset-model-$(date +%Y%m%d%H%M%S)"
```

Once the training is finished the model will be saved in the `output_dir` specified in the command. In this example it's `sd-onepiece-model`. To load the fine-tuned model for inference just pass that path to `StableDiffusionPipeline`

```python
import mindspore as ms
from mindone.diffusers import StableDiffusionPipeline

model_path = "path_to_saved_model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, mindspore_dtype=ms.float16)

image = pipe(prompt="a man in a straw hat")[0][0]
image.save("a-man-in-a-straw-hat.png")
```

Checkpoints only save the unet, so to run inference from a checkpoint, just load the unet

```python
import mindspore as ms
from mindone.diffusers import StableDiffusionPipeline, UNet2DConditionModel

model_path = "path_to_saved_model"
unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-<N>/unet", mindspore_dtype=ms.float16)

pipe = StableDiffusionPipeline.from_pretrained("<initial model>", unet=unet, mindspore_dtype=ms.float16)

image = pipe(prompt="a man in a straw hat")[0][0]
image.save("a-man-in-a-straw-hat.png")
```

#### Training with Min-SNR weighting

We support training with the Min-SNR weighting strategy proposed in [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556) which helps to achieve faster convergence
by rebalancing the loss. In order to use it, one needs to set the `--snr_gamma` argument. The recommended
value when using it is 5.0.

You can find [this project on Weights and Biases](https://wandb.ai/sayakpaul/text2image-finetune-minsnr) that compares the loss surfaces of the following setups:

* Training without the Min-SNR weighting strategy
* Training with the Min-SNR weighting strategy (`snr_gamma` set to 5.0)
* Training with the Min-SNR weighting strategy (`snr_gamma` set to 1.0)

For our small OnePiece dataset, the effects of Min-SNR weighting strategy might not appear to be pronounced, but for larger datasets, we believe the effects will be more pronounced.

Also, note that in this example, we either predict `epsilon` (i.e., the noise) or the `v_prediction`. For both of these cases, the formulation of the Min-SNR weighting strategy that we have used holds.

## Training with LoRA

Low-Rank Adaption of Large Language Models was first introduced by Microsoft in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*.

In a nutshell, LoRA allows adapting pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:

- Previous pretrained weights are kept frozen so that model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114).
- Rank-decomposition matrices have significantly fewer parameters than original model, which means that trained LoRA weights are easily portable.
- LoRA attention layers allow to control to which extent the model is adapted toward new training images via a `scale` parameter.

[cloneofsimo](https://github.com/cloneofsimo) was the first to try out LoRA training for Stable Diffusion in the popular [lora](https://github.com/cloneofsimo/lora) GitHub repository.

With LoRA, it's possible to fine-tune Stable Diffusion on a custom image-caption pair dataset
on consumer NPUs.

### Training

First, you need to set up your development environment as is explained in the [installation section](#installing-the-dependencies). Make sure to set the `MODEL_NAME` and `DATASET_NAME` environment variables. Here, we will use [Stable Diffusion v1-4](https://hf.co/CompVis/stable-diffusion-v1-4) and the [OnePiece dataset](https://huggingface.co/datasets/YaYaB/onepiece-blip-captions).

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**

**___Note: It is quite useful to monitor the training progress by regularly generating sample images during training. [Weights and Biases](https://docs.wandb.ai/quickstart) is a nice solution to easily see generating images during training. All you need to do is to run `pip install wandb` before training to automatically log images.___**

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="YaYaB/onepiece-blip-captions"

python train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --validation_prompt="a man in a straw hat" \
  --output_dir="sd-onepiece-model-lora-$(date +%Y%m%d%H%M%S)"
```

The above command will also run inference as fine-tuning progresses and log the results to local files.

**___Note: When using LoRA we can use a much higher learning rate compared to non-LoRA fine-tuning. Here we use *1e-4* instead of the usual *1e-5*.___**

The final LoRA embedding weights have been uploaded to [sayakpaul/sd-model-finetuned-lora-t4](https://huggingface.co/sayakpaul/sd-model-finetuned-lora-t4). **___Note: [The final weights](https://huggingface.co/sayakpaul/sd-model-finetuned-lora-t4/blob/main/pytorch_lora_weights.bin) are only 3 MB in size, which is orders of magnitudes smaller than the original model.___**

You can check some inference samples that were logged during the course of the fine-tuning process [here](https://wandb.ai/sayakpaul/text2image-fine-tune/runs/q4lc0xsw).

### Inference

Once you have trained a model using above command, the inference can be done simply using the `StableDiffusionPipeline` after loading the trained LoRA weights.  You
need to pass the `output_dir` for loading the LoRA weights which, in this case, is `sd-onepiece-model-lora`.

```python
import mindspore as ms
from mindone.diffusers import StableDiffusionPipeline

model_path = "sayakpaul/sd-model-finetuned-lora-t4"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", mindspore_dtype=ms.float16)
pipe.load_lora_weights(model_path)

prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5)[0][0]
image.save("pokemon.png")
```

If you are loading the LoRA parameters from the Hub and if the Hub repository has
a `base_model` tag (such as [this](https://huggingface.co/sayakpaul/sd-model-finetuned-lora-t4/blob/main/README.md?code=true#L4)), then
you can do:

```py
from huggingface_hub.repocard import RepoCard

lora_model_id = "sayakpaul/sd-model-finetuned-lora-t4"
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = StableDiffusionPipeline.from_pretrained(base_model_id, mindspore_dtype=ms.float16)
...
```

## Stable Diffusion XL

* We support fine-tuning the UNet shipped in [Stable Diffusion XL](https://huggingface.co/papers/2307.01952) via the `train_text_to_image_sdxl.py` script. Please refer to the docs [here](./README_sdxl.md).
