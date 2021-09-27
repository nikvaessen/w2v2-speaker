################################################################################
#
# Utility functions for collating a batch with different number of frames in
# each sample.
#
# The functions assume an input dimensionality of
#
# [NUM_FEATURES, NUM_FRAMES] for MFCC-like input
# or
# [NUM_FRAMES] for wave-like input
#
# This will result in batches with respective dimensionality
# [NUM_SAMPLES, NUM_FEATURES, NUM_FRAMES] or [NUM_SAMPLES, NUM_FRAMES]
#
# Author(s): Nik Vaessen
################################################################################

from typing import List, Callable, Optional

import torch as t

from torch.nn import (
    ConstantPad1d,
    ConstantPad2d,
)


################################################################################
# private utility functions


def _determine_max_num_frames(
    samples: List[t.Tensor], frame_dim: int = 0, feature_dim: Optional[int] = None
):
    if len(samples) <= 0:
        raise ValueError("expected non-empty list")
    if frame_dim == feature_dim:
        raise ValueError("frame_dim and feature_dim cannot be equal")
    if not (0 <= frame_dim <= 1):
        raise ValueError(f"frame_dim should be either 0 or 1, not {frame_dim=}")
    if feature_dim is not None and not (0 <= feature_dim <= 1):
        raise ValueError(f"feature_dim should be either 0 or 1, not {feature_dim=}")

    # assume [NUM_FRAMES] or [NUM_FRAMES, NUM_FEATURES]
    max_frames = -1
    num_features = None

    for idx, sample in enumerate(samples):
        num_dim = len(sample.shape)

        if not (num_dim == 1 or num_dim == 2):
            raise ValueError(
                "only 1 or 2-dimensional samples are supported."
                f"Received sample with shape {sample.shape}"
            )
        elif num_dim == 2:
            if feature_dim is None:
                raise ValueError(
                    "padding a 2 dimensional tensor requires setting `feature_dim`"
                )
            if idx == 0:
                num_features = sample.shape[feature_dim]
            elif num_features != sample.shape[feature_dim]:
                raise ValueError(
                    "list has inconsistent number of features. "
                    f"Received at least {num_features} and {sample.shape[0]}"
                )

        num_frames = sample.shape[frame_dim]

        if num_frames > max_frames:
            max_frames = num_frames

    return max_frames


def _generic_append_padding(
    samples: List[t.Tensor],
    padding_init: Callable[[int, int, int], t.nn.Module],
    frame_dim: int = 0,
    feature_dim: Optional[int] = 1,
):
    max_frames = _determine_max_num_frames(samples, frame_dim, feature_dim)

    padded_samples = []

    for sample in samples:
        num_dim = len(sample.shape)
        num_frames = sample.shape[frame_dim]

        padded_sample = padding_init(num_dim, num_frames, max_frames)(sample)

        padded_samples.append(padded_sample)

    return t.stack(padded_samples)


################################################################################
# constant collating


def collate_append_constant(
    samples: List[t.Tensor],
    value: float = 0,
    frame_dim: int = 0,
    feature_dim: Optional[int] = None,
):
    def padding_init(num_dim: int, num_frames: int, max_frames: int, v=value):
        padding_right = max_frames - num_frames

        if num_dim == 1:
            return ConstantPad1d((0, padding_right), v)
        else:
            if frame_dim == 0:
                return ConstantPad2d((0, 0, 0, padding_right), v)
            elif frame_dim == 1:
                return ConstantPad2d((0, padding_right, 0, 0), v)
            else:
                raise ValueError("frame_dim can only be 0 or 1")

    return _generic_append_padding(samples, padding_init, frame_dim, feature_dim)


################################################################################
# reflective collating


def collate_append_reflection(samples: List[t.Tensor]):
    raise NotImplemented()


################################################################################
# repetitive collating


def collate_append_repeat(samples: List[t.Tensor]):
    raise NotImplemented()
