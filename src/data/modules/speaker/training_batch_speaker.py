################################################################################
#
# Dataclass of batch which is output of dataloader.
#
# This batch simply has a single integer label representing a speaker ID
# for each sample in the batch.
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
class SpeakerClassificationDataSample:
    # a unique identifier for this ground_truth and input tensor pairing
    key: str

    # integer value for class of sample
    ground_truth: int

    # tensor of floats with shape depending on the particular task
    # shape is most likely [NUM_FRAMES, NUM_FEATURES]
    network_input: t.Tensor

    # additional information which does not/cannot/should not
    # be (easily) collated
    side_info: Optional[BatchDebugInfo]


@dataclasses.dataclass
class SpeakerClassificationDataBatch:
    # the number of samples this batch contains
    batch_size: int

    # list of strings with length BATCH_SIZE where each index matches
    # a unique identifier to the ground_truth or input tensor at the
    # particular batch dimension
    keys: List[str]

    # tensor of floats with shape [BATCH_SIZE, NUM_FRAMES, NUM_FEATURES]
    network_input: t.Tensor

    # tensor of integers with shape [BATCH_SIZE]
    ground_truth: t.Tensor

    # the side information per sample based on a mapping
    # between a key at a particular index of `keys` and the corresponding
    # network_input at that index of the BATCH_SIZE dimension
    side_info: Dict[str, BatchDebugInfo]

    def __len__(self):
        return self.batch_size

    def to(self, device: torch.device) -> "SpeakerClassificationDataBatch":
        return SpeakerClassificationDataBatch(
            self.batch_size,
            self.keys,
            self.network_input.to(device),
            self.ground_truth.to(device),
            self.side_info,
        )

    @staticmethod
    def default_collate_fn(
        lst: List[SpeakerClassificationDataSample],
    ) -> "SpeakerClassificationDataBatch":
        batch_size = len(lst)
        keys = default_collate([sample.key for sample in lst])
        network_input = default_collate([sample.network_input for sample in lst])
        ground_truth = t.squeeze(
            default_collate([sample.ground_truth for sample in lst])
        )
        side_info = {sample.key: sample.side_info for sample in lst}

        return SpeakerClassificationDataBatch(
            batch_size=batch_size,
            keys=keys,
            network_input=network_input,
            ground_truth=ground_truth,
            side_info=side_info,
        )

    @staticmethod
    def pad_right_collate_fn(
        lst: List[SpeakerClassificationDataSample],
    ) -> "SpeakerClassificationDataBatch":
        batch_size = len(lst)
        keys = default_collate([sample.key for sample in lst])
        network_input = collate_append_constant(
            [sample.network_input for sample in lst], frame_dim=0, feature_dim=1
        )
        ground_truth = t.squeeze(
            default_collate([sample.ground_truth for sample in lst])
        )
        side_info = {sample.key: sample.side_info for sample in lst}

        return SpeakerClassificationDataBatch(
            batch_size=batch_size,
            keys=keys,
            network_input=network_input,
            ground_truth=ground_truth,
            side_info=side_info,
        )


################################################################################
# samples and batches for paired training


@dataclasses.dataclass
class PairedSpeakerClassificationDataSample:
    # a unique identifier for the pair of network inputs
    primary_key: str
    secondary_key: str

    # a pair of inputs representing 2 audio files
    # shape is most likely [NUM_FRAMES, NUM_FEATURES] with potentially different
    # numbers of frames primary and secondary input
    primary_input: t.Tensor
    secondary_input: t.Tensor

    # integer value for class of sample
    ground_truth: int

    # additional information which does not/cannot/should not
    # be (easily) collated
    side_info: Optional[BatchDebugInfo]


@dataclasses.dataclass
class PairedSpeakerClassificationDataBatch:
    # the number of samples this batch contains
    batch_size: int

    # list of strings with length BATCH_SIZE where each index matches
    # a unique identifier to the ground_truth and primary tensor at the
    # particular batch dimension
    primary_keys: List[str]

    # list of strings with length BATCH_SIZE where each index matches
    # a unique identifier to the ground_truth and secondary input tensor at the
    # particular batch dimension
    secondary_keys: List[str]

    # tensor of floats with shape [BATCH_SIZE, NUM_FRAMES, NUM_FEATURES]
    primary_network_input: t.Tensor

    # tensor of floats with shape [BATCH_SIZE, NUM_FRAMES, NUM_FEATURES]
    # NUM_FEATURES might differ between primary and secondary network input
    secondary_network_input: t.Tensor

    # tensor of integer values (0 or 1) with shape [BATCH_SIZE]
    ground_truth: t.Tensor

    # the side information per sample based on a mapping
    # between a key at a particular index of `keys` and the corresponding
    # network_input at that index of the BATCH_SIZE dimension
    side_info: Dict[str, BatchDebugInfo]

    def __len__(self):
        return self.batch_size

    def to(self, device: torch.device) -> "PairedSpeakerClassificationDataBatch":
        return PairedSpeakerClassificationDataBatch(
            self.batch_size,
            self.primary_keys,
            self.secondary_keys,
            self.primary_network_input.to(device),
            self.secondary_network_input.to(device),
            self.ground_truth.to(device),
            self.side_info,
        )

    @staticmethod
    def default_collate_fn(
        lst: List[PairedSpeakerClassificationDataSample],
    ) -> "PairedSpeakerClassificationDataBatch":
        batch_size = len(lst)

        primary_keys = default_collate([sample.primary_key for sample in lst])
        secondary_keys = default_collate([sample.secondary_key for sample in lst])

        primary_input = default_collate(
            [sample.primary_input.squeeze() for sample in lst]
        )
        secondary_input = default_collate(
            [sample.secondary_input.squeeze() for sample in lst]
        )

        ground_truth = t.squeeze(
            default_collate([sample.ground_truth for sample in lst])
        )
        side_info = {sample.primary_key: sample.side_info for sample in lst}

        return PairedSpeakerClassificationDataBatch(
            batch_size=batch_size,
            primary_keys=primary_keys,
            secondary_keys=secondary_keys,
            primary_network_input=primary_input,
            secondary_network_input=secondary_input,
            ground_truth=ground_truth,
            side_info=side_info,
        )

    @staticmethod
    def pad_right_collate_fn(
        lst: List[PairedSpeakerClassificationDataSample],
    ) -> "PairedSpeakerClassificationDataBatch":
        batch_size = len(lst)

        primary_keys = default_collate([sample.primary_key for sample in lst])
        secondary_keys = default_collate([sample.secondary_key for sample in lst])

        primary_input = collate_append_constant(
            [sample.primary_input for sample in lst], frame_dim=1, feature_dim=0
        )
        secondary_input = collate_append_constant(
            [sample.secondary_input for sample in lst], frame_dim=1, feature_dim=0
        )

        ground_truth = t.squeeze(
            default_collate([sample.ground_truth for sample in lst])
        )
        side_info = {sample.primary_key: sample.side_info for sample in lst}

        return PairedSpeakerClassificationDataBatch(
            batch_size=batch_size,
            primary_keys=primary_keys,
            secondary_keys=secondary_keys,
            primary_network_input=primary_input.squeeze(),
            secondary_network_input=secondary_input.squeeze(),
            ground_truth=ground_truth,
            side_info=side_info,
        )
