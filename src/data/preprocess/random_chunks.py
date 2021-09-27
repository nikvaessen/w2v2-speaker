################################################################################
#
# Select a
#
# Author(s): Nik Vaessen
################################################################################

import pathlib
import random
from typing import Union, List

import torch

from enum import Enum

import torch as t
import torchaudio

from src.data.common import DebugWriter
from src.data.preprocess.base import Preprocessor
from src.data.modules.speaker.training_batch_speaker import (
    SpeakerClassificationDataSample,
    BatchDebugInfo,
)
from src.util import debug_tensor_content

################################################################################
# implementation of the selector


class AudioChunkDebugWriter(DebugWriter):
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def write(self, tensor: t.Tensor, save_dir: pathlib.Path, idx: int):
        debug_tensor_content(
            tensor, f"{idx:03d}_randomly_selected_audio_chunk", save_dir
        )
        torchaudio.save(
            str(save_dir / f"{idx:03d}_randomly_selected_audio_chunk.wav"),
            tensor,
            self.sample_rate,
        )


class SelectionStrategy(str, Enum):
    start = "start"
    end = "end"
    random = "random"
    random_contiguous = "random_contiguous"
    contiguous = "contiguous"


class AudioChunkSelector(Preprocessor):
    def __init__(
        self,
        selection_strategy: SelectionStrategy,
        desired_chunk_length_sec: float,
        sample_rate: int = 16000,
        yield_all_contiguous: bool = False,
    ):
        """
        Randomly select a subsample of a audio tensor where the last dimension
        contains the audio observations

        :param selection_strategy: how to select the subsample
        :param desired_chunk_length_sec: the desired length of the subsample in seconds
        :param sample_rate: the sample rate of the audio
        """
        super().__init__()

        if selection_strategy == SelectionStrategy.start:
            self.fn = self._start_select
        elif selection_strategy == SelectionStrategy.end:
            self.fn = self._end_select
        elif selection_strategy == SelectionStrategy.random:
            self.fn = self._random_select
        elif selection_strategy == SelectionStrategy.random_contiguous:
            self.fn = self._random_contiguous_select
        elif selection_strategy == SelectionStrategy.contiguous:
            self.fn = self._contiguous_select
        else:
            raise ValueError(f"unknown selection strategy {selection_strategy}")

        self.chunk_size = round(sample_rate * desired_chunk_length_sec)
        self.sample_rate = sample_rate
        self.yield_all_contiguous = yield_all_contiguous

    def process(
        self, sample: SpeakerClassificationDataSample
    ) -> Union[SpeakerClassificationDataSample, List[SpeakerClassificationDataSample]]:
        chunked_wavs = [c for c in self.fn(sample.network_input)]

        if len(chunked_wavs) == 1:
            sample.network_input = chunked_wavs[0]

            if sample.side_info is not None:
                sample.side_info.pipeline_progress.append(
                    (sample.network_input, self.init_debug_writer())
                )

            return sample

        elif len(chunked_wavs) > 1:
            samples = []

            for idx, selected_wav in enumerate(chunked_wavs):
                new_network_input = selected_wav

                if sample.side_info is not None:
                    new_side_info = BatchDebugInfo(
                        original_tensor=sample.side_info.original_tensor,
                        pipeline_progress=list(sample.side_info.pipeline_progress)
                        + [(new_network_input, self.init_debug_writer())],
                        meta=sample.side_info.meta,
                    )
                else:
                    new_side_info = None

                new_sample = SpeakerClassificationDataSample(
                    key=sample.key + f"/chunk{idx}",
                    network_input=new_network_input,
                    ground_truth=sample.ground_truth,
                    side_info=new_side_info,
                )

                samples.append(new_sample)

            return samples

        else:
            raise ValueError("unable to select at least one chunk")

    def init_debug_writer(self):
        return AudioChunkDebugWriter(self.sample_rate)

    def _start_select(self, wav_tensor: torch.Tensor):
        yield wav_tensor[..., : self.chunk_size]

    def _end_select(self, wav_tensor: torch.Tensor):
        yield wav_tensor[..., -self.chunk_size :]

    def _random_select(self, wav_tensor: torch.Tensor):
        num_samples = wav_tensor.shape[-1]

        if self.chunk_size > num_samples:
            yield wav_tensor[..., :]
        else:
            start = random.randint(0, num_samples - self.chunk_size - 1)
            end = start + self.chunk_size
            yield wav_tensor[..., start:end]

    def _random_contiguous_select(self, wav_tensor: torch.Tensor):
        num_samples = wav_tensor.shape[-1]

        num_possible_chunks = num_samples // self.chunk_size
        selected_chunk = random.randint(0, num_possible_chunks - 1)

        start = selected_chunk * self.chunk_size
        end = start + self.chunk_size

        yield wav_tensor[..., start:end]

    def _contiguous_select(self, wav_tensor: torch.Tensor):
        num_samples = wav_tensor.shape[-1]
        num_possible_chunks = num_samples // self.chunk_size

        for selected_chunk in range(num_possible_chunks):
            start = selected_chunk * self.chunk_size
            end = start + self.chunk_size

            yield wav_tensor[..., start:end]
