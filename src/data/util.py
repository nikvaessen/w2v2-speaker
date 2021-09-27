################################################################################
#
# Utility functions related to data i/o
#
# Author(s): Nik Vaessen
################################################################################

import pathlib

import torchaudio

import numpy as np
import torch as t

################################################################################
# read audio from wav file into a tensor


def load_raw_audio(path: pathlib.Path) -> t.Tensor:
    """
    Load the raw audio file at the specified path and return it as a tensor
    with shape [1, num_samples] with floating values between -1 and 1

    :param path: the path to the audio value
    :return: a tensor of shape [1, num_samples] of the raw audio
    """
    tensor, sample_rate = torchaudio.load(str(path))

    if sample_rate != 16000:
        raise ValueError(
            f"audio file {path} is expected to have a sampling"
            f" rate of 16000 while actually being {sample_rate}"
        )

    return tensor


################################################################################
# read/save tensors


def load_tensor(path: pathlib.Path, device=t.device("cpu")) -> t.Tensor:
    return t.load(path, map_location=device)


def save_tensor(embedding: t.Tensor, save_path: pathlib.Path):
    save_path.parent.mkdir(exist_ok=True, parents=True)
    t.save(embedding, str(save_path))


################################################################################
# hacky way to create a None tensor


def create_nan_tensor():
    return t.Tensor([np.nan])


def is_nan_tensor(tensor: t.Tensor):
    return t.all(t.isnan(tensor)).item()


################################################################################
# check if a value can cause nan/inf


def tensor_has_inf(tensor: t.Tensor):
    return t.any(t.isinf(tensor)).item()


def tensor_has_nan(tensor: t.Tensor):
    return t.any(t.isnan(tensor)).item()


def is_invalid_tensor(tensor: t.Tensor):
    return tensor_has_inf(tensor) or tensor_has_inf(tensor)
