# Adapted from
# https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/eval/fvd/videogpt/fvd.py

import math
import os
import shlex
import subprocess

import numpy as np

import mindspore as ms
from mindspore import Tensor, mint, ops

try:
    import torch
except ImportError:
    print(
        "For the first-time running, torch is required to load torchscript model and convert to onnx, but import torch leads to an ImportError!"
    )


def load_i3d_pretrained():
    i3D_WEIGHTS_URL = (
        "https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI"
    )
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i3d_pretrained_400.pt")
    ms_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i3d_pretrained_400.ckpt")
    if not os.path.exists(ms_filepath):
        if not os.path.exists(filepath):
            print(f"preparing for download {i3D_WEIGHTS_URL}, you can download it by yourself.")
            subprocess.run(shlex.split(f"wget {i3D_WEIGHTS_URL} -O {filepath}"), shell=False)
        # convert torch ckpt to mindspore ckpt
        state_dict = torch.load_state_dict(torch.load(filepath))
        raise ValueError("Not converted")
    from .ms_i3d import InceptionI3d

    model = InceptionI3d(400, in_channels=3)
    state_dict = ms.load_checkpoint(ms_filepath)
    m, u = ms.load_param_into_net(model, state_dict)
    print("net param not load: ", m, len(m))
    print("ckpt param not load: ", u, len(u))

    model.set_train(False)
    for param in model.get_parameters():
        param.requires_grad = False
    return model


def preprocess_single(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.transpose(0, 3, 1, 2).float() / 255.0  # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = ops.interpolate(video, size=target_size, mode="bilinear", align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start : h_start + resolution, w_start : w_start + resolution]
    video = video.transpose(1, 0, 2, 3)  # CTHW

    video -= 0.5

    return video


def preprocess(videos, target_resolution=224):
    # we should tras videos in [0-1] [b c t h w] as th.float
    # -> videos in {0, ..., 255} [b t h w c] as np.uint8 array
    # b c t h w -> b t h w c
    videos = videos.transpose(0, 2, 3, 4, 1)
    videos = (videos * 255).numpy().astype(np.uint8)

    b, t, h, w, c = videos.shape
    videos = Tensor(videos)
    videos = mint.stack([preprocess_single(video, target_resolution) for video in videos])
    return videos * 2  # [-0.5, 0.5] -> [-1, 1]


def get_fvd_logits(videos, i3d, bs=10):
    videos = preprocess(videos)
    embeddings = get_logits(i3d, videos, bs=bs)
    return embeddings


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    s, u, v = ops.svd(mat)
    si = mint.where(s < eps, s, mint.sqrt(s))
    return mint.matmul(mint.matmul(u, ops.diag(si)), v.t())


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = mint.matmul(sqrt_sigma, mint.matmul(sigma_v, sqrt_sigma))
    return mint.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
    """Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.ndim < 2:
        m = m.view(1, -1)
    if not rowvar and m.shape[0] != 1:
        m = m.t()

    fact = 1.0 / (m.shape[1] - 1)  # unbiased estimate
    m -= mint.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def frechet_distance(x1, x2):
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = x1.mean(dim=0), x2.mean(dim=0)
    sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)
    mean = mint.sum((m - m_w) ** 2)
    if x1.shape[0] > 1:
        sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
        trace = mint.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component
        fd = trace + mean
    else:
        fd = np.real(mean)
    return float(fd)


def get_logits(i3d, videos, bs=10):
    # assert videos.shape[0] % 16 == 0
    logits = []
    for i in range(0, videos.shape[0], bs):
        batch = videos[i : i + bs]
        # logits.append(i3d.module.extract_features(batch)) # wrong
        logits.append(i3d(batch))  # right
    logits = mint.cat(logits, dim=0)
    return logits
