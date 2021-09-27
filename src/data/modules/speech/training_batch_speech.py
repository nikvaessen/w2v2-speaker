################################################################################
#
# Dataclass of batch which is output of dataloader
#
# Author(s): Nik Vaessen
################################################################################

import dataclasses

from typing import List, Dict, Optional

import torch
import torch as t

from torch.utils.data._utils.collate import default_collate

from src.data.collating import collate_append_constant
from src.data.common import BatchDebugInfo

################################################################################
# data sample and data batch classes


@dataclasses.dataclass
class SpeechRecognitionDataSample:
    # a unique identifier for this ground_truth and input tensor pairing
    key: str

    # sequence of integer values. '0' should represent a blank
    # and >= 1 should represent a letter or symbol. Shape should be
    # [SEQUENCE_LENGTH,]
    ground_truth: t.Tensor

    # transcription in string format
    ground_truth_string: str

    # tensor of floats with shape depending on the particular task
    # shape is most likely [NUM_FRAMES, NUM_FEATURES] (for mfcc-like input)
    # or [NUM_FRAMES,] for wav-like input.
    network_input: t.Tensor

    # length of input before padding (equal to NUM_FRAMES)
    input_length: int

    # the length of the `ground_truth` tensor
    ground_truth_sequence_length: t.Tensor

    # additional information which does not/cannot/should not
    # be (easily) collated
    side_info: Optional[BatchDebugInfo]


@dataclasses.dataclass
class SpeechRecognitionDataBatch:
    # the number of samples this batch contains
    batch_size: int

    # list of strings with length BATCH_SIZE where each index matches
    # a unique identifier to the ground_truth or input tensor at the
    # particular batch dimension
    keys: List[str]

    # tensor of floats with shape [BATCH_SIZE, NUM_FRAMES, NUM_FEATURES]
    network_input: t.Tensor

    # list of length of audio before padding
    input_lengths: List[int]

    # tensor of transcription strings with length [BATCH_SIZE, MAX_SEQUENCE_LENGTH]
    ground_truth: t.Tensor

    # transcriptions in string format
    ground_truth_strings: List[str]

    # tensor of shape [BATCH_SIZE,] where each value contains the
    # sequence length (before padding was added) of the respective batch dimension.
    ground_truth_sequence_length: t.Tensor

    # the side information per sample based on a mapping
    # between a key at a particular index of `keys` and the corresponding
    # network_input at that index of the BATCH_SIZE dimension
    side_info: Dict[str, BatchDebugInfo]

    def __len__(self):
        return self.batch_size

    def to(self, device: torch.device) -> "SpeechRecognitionDataBatch":
        return SpeechRecognitionDataBatch(
            self.batch_size,
            self.keys,
            self.network_input.to(device),
            self.input_lengths,
            self.ground_truth.to(device),
            self.ground_truth_strings,
            self.ground_truth_sequence_length,
            self.side_info,
        )


################################################################################
# function(s) to collate a collection of training batches


def default_collate_fn(
    lst: List[SpeechRecognitionDataSample],
) -> SpeechRecognitionDataBatch:
    try:
        batch_size = len(lst)
        keys = default_collate([sample.key for sample in lst])
        network_input = collate_append_constant(
            [sample.network_input.squeeze() for sample in lst],
        )
        input_lengths = [sample.input_length for sample in lst]
        ground_truth = t.squeeze(
            collate_append_constant([sample.ground_truth for sample in lst])
        )
        ground_truth_strings = [sample.ground_truth_string for sample in lst]
        ground_truth_sequence_length = t.squeeze(
            default_collate([sample.ground_truth_sequence_length for sample in lst])
        )
        side_info = {sample.key: sample.side_info for sample in lst}

        return SpeechRecognitionDataBatch(
            batch_size,
            keys,
            network_input,
            input_lengths,
            ground_truth,
            ground_truth_strings,
            ground_truth_sequence_length,
            side_info,
        )
    except:
        print(len(lst))
        print(lst[0])
