################################################################################
#
# Base API for preprocessors
#
# Author(s): Nik Vaessen
################################################################################

import pathlib

from typing import Union, List

import librosa
import torch as t
import torchaudio
import seaborn

from matplotlib import pyplot as plt
from speechbrain.lobes.features import Fbank

from src.data.common import DebugWriter
from src.data.preprocess.base import Preprocessor
from src.data.modules.speaker.training_batch_speaker import (
    SpeakerClassificationDataSample,
)
from src.util import debug_tensor_content

################################################################################
# base preprocessor


class FilterBankDebugWriter(DebugWriter):
    def write(self, tensor: t.Tensor, save_dir: pathlib.Path, idx: int):
        debug_tensor_content(tensor, f"{idx:03d}_filterbank_features", save_dir)

        # make a plot of the filterbank values
        heatmap = seaborn.heatmap(tensor.cpu().numpy())
        fig = heatmap.get_figure()
        fig.savefig(str(save_dir / f"{idx:03d}_filterbank_features.png"))
        plt.clf()

        # convert back to audio
        a1 = tensor.numpy().transpose()
        a1 = librosa.db_to_amplitude(a1)
        a1 = librosa.feature.inverse.mel_to_audio(
            a1,
            n_fft=400,
            fmin=0,
            fmax=8000,
            hop_length=160,
            win_length=16 * 25,
            center=False,
            power=1,
            n_iter=10,
        )

        torchaudio.save(
            save_dir / f"{idx:03d}_filterbank_features.wav",
            t.Tensor(a1)[None, :],
            16000,
        )


class FilterBank(Preprocessor):
    def __init__(self, n_mels: int = 40):
        self.fb = Fbank(n_mels=n_mels)

    def process(
        self, sample: SpeakerClassificationDataSample
    ) -> Union[SpeakerClassificationDataSample, List[SpeakerClassificationDataSample]]:
        # expects an audio file of shape [1, NUM_AUDIO_SAMPLES] and converts
        # to [1, NUM_FRAMES, N_MELS] which is squeezed to [NUM_FRAMES, N_MELS]
        sample.network_input = self.fb(sample.network_input).squeeze()

        if sample.side_info is not None:
            sample.side_info.pipeline_progress.append(
                (sample.network_input, self.init_debug_writer())
            )

        return sample

    def init_debug_writer(
        self,
    ):
        return FilterBankDebugWriter()
