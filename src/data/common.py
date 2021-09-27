################################################################################
#
# A collection of common data classes
#
# Author(s): Nik Vaessen
################################################################################

import pathlib

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import torch as t

################################################################################
#


@dataclass
class WebDataSetShardConfig:
    samples_per_shard: int
    use_gzip_compression: bool
    shuffle_shards: bool
    queue_size: int


@dataclass
class SpeakerDataLoaderConfig:
    num_workers: int
    train_batch_size: int
    val_batch_size: int
    test_batch_size: int
    pin_memory: bool


@dataclass
class SpeechDataLoaderConfig:
    num_workers: int
    train_max_num_samples: int
    val_batch_size: int
    test_batch_size: int
    pin_memory: bool


################################################################################
#


@dataclass
class DebugWriter:
    @abstractmethod
    def write(self, tensor: t.Tensor, save_dir: pathlib.Path, idx: int):
        pass


@dataclass
class BatchDebugInfo:
    # the original tensor which should be easily converted to
    # e.g an image/audio file
    original_tensor: t.Tensor

    # a list containing the progression steps from the original_tensor
    # to the network_input tensor accompanied with a class which can be
    # used to write debug output to a particular folder
    pipeline_progress: List[
        Tuple[
            t.Tensor,
            DebugWriter,
        ]
    ]

    # optional (untyped) dataset specific information
    # about the data sample
    meta: Optional[Dict[Any, Any]]
