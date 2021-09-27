################################################################################
#
# Different augmentation which can be applied to audio (in time domain)
#
# Author(s): Nik Vaessen
################################################################################

import pathlib

from abc import abstractmethod
from typing import List, Union

import torchaudio

import numpy as np
import torch as t
import augment as wavaugment
import webdataset as wds

from speechbrain.lobes.augment import TimeDomainSpecAugment
from torch.utils.data import DataLoader

from src.data.preprocess.base import Preprocessor
from src.data.modules.speaker.training_batch_speaker import (
    SpeakerClassificationDataSample,
    BatchDebugInfo,
    DebugWriter,
)
from src.util import debug_tensor_content

################################################################################
# debug writers


class AugmentationDebugWriter(DebugWriter):
    def __init__(self, name: str, sample_rate: int):
        self.name = name
        self.sample_rate = sample_rate

    def write(self, tensor: t.Tensor, save_dir: pathlib.Path, idx: int):
        debug_tensor_content(
            tensor,
            f"{idx:03d}_augmentation_{self.name}",
            save_dir,
        )
        torchaudio.save(
            str(save_dir / f"{idx:03d}_augmentation_{self.name}.wav"),
            tensor,
            self.sample_rate,
        )


################################################################################
# spec augmentation from speechbrain


class SpecAugmentTimeDomain(Preprocessor):
    def __init__(self, speeds: List[int], sample_rate: int):
        self.fn = TimeDomainSpecAugment(speeds=speeds, sample_rate=sample_rate)

        self.name = "speed" + "_".join([str(s) for s in speeds])
        self.sample_rate = sample_rate

    def process(self, x: t.Tensor) -> [t.Tensor]:
        # assume a tensor of shape [BATCH_SIZE, NUM_AUDIO_FRAMES]
        bs = x.shape[0]

        return self.fn(x, t.ones(bs))

    def init_debug_writer(self):
        return AugmentationDebugWriter(self.name, self.sample_rate)


################################################################################
# processor for augmenting with the `WavAugment` library


class Augmenter(Preprocessor):
    def __init__(
        self,
        augmenters: List["WavAugmentWrapper"],
        stack_augmentations: bool,
        yield_intermediate_augmentations: bool,
        yield_unaugmented: bool,
    ):
        self.augmenters = augmenters

        if not stack_augmentations and not yield_intermediate_augmentations:
            raise ValueError(
                "augmenter must at least stack "
                "augmentations or yield intermediate augmentations"
            )

        self.stack_augmentations = stack_augmentations
        self.yield_intermediate_augmentations = yield_intermediate_augmentations
        self.yield_unaugmented = yield_unaugmented

    def process(
        self, sample: SpeakerClassificationDataSample
    ) -> Union[SpeakerClassificationDataSample, List[SpeakerClassificationDataSample]]:
        current_sample: SpeakerClassificationDataSample = sample
        intermediary_samples = []

        if self.yield_unaugmented:
            intermediary_samples.append(current_sample)

        for augmenter in self.augmenters:
            new_network_input = augmenter.process(current_sample.network_input)

            if current_sample.side_info is not None:
                new_side_info = BatchDebugInfo(
                    original_tensor=current_sample.side_info.original_tensor,
                    pipeline_progress=list(current_sample.side_info.pipeline_progress)
                    + [(new_network_input, augmenter.init_debug_writer())],
                    meta=current_sample.side_info.meta,
                )
            else:
                new_side_info = None

            new_sample = SpeakerClassificationDataSample(
                key=current_sample.key + f"/{augmenter.name}",
                network_input=new_network_input,
                ground_truth=current_sample.ground_truth,
                side_info=new_side_info,
            )

            if self.yield_intermediate_augmentations:
                intermediary_samples.append(new_sample)

            if self.stack_augmentations:
                current_sample = new_sample

        if not self.yield_intermediate_augmentations:
            return current_sample
        else:
            return intermediary_samples

    def init_debug_writer(
        self,
    ):
        # this is handled by each WavAugmentWrapper individually
        raise NotImplementedError()


################################################################################
# define a class for each augmentation provided by WavAugment library


class WavAugmentWrapper:
    def __init__(self, sample_rate: int, name: str):
        self.sample_rate = sample_rate
        self.name = name

    def process(self, x: t.Tensor) -> [t.Tensor]:
        return self._get_chain().apply(
            x,
            src_info={"rate": self.sample_rate},
            target_info={"channels": 1, "rate": self.sample_rate},
        )

    @abstractmethod
    def _get_chain(self) -> wavaugment.EffectChain:
        pass

    def init_debug_writer(
        self,
    ):
        return AugmentationDebugWriter(self.name, self.sample_rate)


################################################################################
# augment speed of audio


class UniformSpeedAugment(WavAugmentWrapper):
    def __init__(
        self, sample_rate: int, min_speed_factor: float, max_speed_factor: float
    ):
        super(UniformSpeedAugment, self).__init__(sample_rate, "uniform_speed")

        self.min_speed = min_speed_factor
        self.max_speed = max_speed_factor

        self._chain: wavaugment.EffectChain = (
            wavaugment.EffectChain().speed(self._random_float).rate(sample_rate)
        )

    def _random_float(self):
        return np.random.uniform(self.min_speed, self.max_speed)

    def _get_chain(self) -> wavaugment.EffectChain:
        return self._chain


class ChoiceSpeedAugment(WavAugmentWrapper):
    def __init__(self, sample_rate: int, possible_speed_factors: List[float]):
        super(ChoiceSpeedAugment, self).__init__(sample_rate, "choice_speed")

        self.choices = possible_speed_factors

        self._chain: wavaugment.EffectChain = (
            wavaugment.EffectChain().speed(self._random_float).rate(sample_rate)
        )

    def _random_float(self):
        return np.random.choice(self.choices)

    def _get_chain(self) -> wavaugment.EffectChain:
        return self._chain


################################################################################
# randomly replace parts of the audio with (with silence)


class TimeDropoutAugment(WavAugmentWrapper):
    def __init__(
        self,
        sample_rate: int,
        max_dropout_length_seconds: float,
        min_drop_count: int,
        max_drop_count: int,
    ):
        super(TimeDropoutAugment, self).__init__(sample_rate, "time_dropout")

        self.max_dropout_length_seconds = max_dropout_length_seconds
        self.min_drops = min_drop_count
        self.max_drops = max_drop_count

    def _get_random_count(self):
        return np.random.randint(self.min_drops, self.max_drops + 1)

    def _get_chain(self) -> wavaugment.EffectChain:
        chain = wavaugment.EffectChain()

        for i in range(self._get_random_count()):
            chain = chain.time_dropout(max_seconds=self.max_dropout_length_seconds)

        return chain


################################################################################
# randomly remove certain frequencies from the audio


class SpecAugmentBand:
    def __init__(self, sampling_rate, scalar):
        self.sampling_rate = sampling_rate
        self.scalar = scalar

    @staticmethod
    def freq2mel(f):
        return 2595.0 * np.log10(1 + f / 700)

    @staticmethod
    def mel2freq(m):
        return (10.0 ** (m / 2595.0) - 1) * 700

    def __call__(self):
        F = 27.0 * self.scalar

        melfmax = self.freq2mel(self.sampling_rate / 2)
        meldf = np.random.uniform(0, melfmax * F / 256.0)
        melf0 = np.random.uniform(0, melfmax - meldf)

        low = self.mel2freq(melf0)
        high = self.mel2freq(melf0 + meldf)

        return f"{high}-{low}"


class FrequencyDropoutAugment(WavAugmentWrapper):
    def __init__(
        self,
        sample_rate: int,
        min_drop_count: int,
        max_drop_count: int,
        band_scaling: float,
    ):
        super(FrequencyDropoutAugment, self).__init__(sample_rate, "frequency_dropout")

        self.min_drops = min_drop_count
        self.max_drops = max_drop_count
        self.band_scaling = band_scaling

    def _get_random_count(self):
        return np.random.randint(self.min_drops, self.max_drops + 1)

    def _get_chain(self) -> wavaugment.EffectChain:
        chain = wavaugment.EffectChain()

        for i in range(self._get_random_count()):
            chain = chain.sinc(
                "-a", "120", SpecAugmentBand(self.sample_rate, self.band_scaling)
            )

        return chain


################################################################################
# Randomly add certain noise to the audio


class ChoiceRandomNoiseAugment(WavAugmentWrapper):
    def __init__(self, sample_rate: int, snr_choices: List[int]):
        super().__init__(sample_rate, "uniform_noise")

        self.snr_choices = snr_choices

    @staticmethod
    def _noise_generator(x: t.Tensor):
        return t.zeros_like(x).uniform_()

    def _random_snr(self):
        return np.random.choice(self.snr_choices)

    def _get_chain(self) -> wavaugment.EffectChain:
        raise NotImplementedError()

    def process(self, x: t.Tensor) -> [t.Tensor]:
        chain = wavaugment.EffectChain().additive_noise(
            lambda: self._noise_generator(x), snr=self._random_snr()
        )

        return chain.apply(
            x,
            src_info={"rate": self.sample_rate},
            target_info={"channels": 1, "rate": self.sample_rate},
        )


################################################################################
# add RIR point source noise


def no_split(urls):
    return urls


class ChoiceRirsNoiseAugment(WavAugmentWrapper):
    def __init__(
        self, sample_rate: int, snr_choices: List[int], shards_folder: pathlib.Path
    ):
        super().__init__(sample_rate, "rirs_background_noise")

        if isinstance(shards_folder, str):
            shards_folder = pathlib.Path(shards_folder)

        self.snr_choices = snr_choices
        self.shards_folder = shards_folder

        self.dataloader = self._generate_dataloader()
        self._iter = None

    def _generate_dataloader(self):
        ds = (
            wds.WebDataset(
                [
                    str(x)
                    for x in self.shards_folder.iterdir()
                    if "pointsource_noises.tar" in x.name
                ],
                nodesplitter=no_split,
                splitter=no_split,
            )
            .decode(wds.torch_audio)
            .shuffle(10)
            .repeat()
        )

        return ds
        # return DataLoader(ds, num_workers=0, batch_size=None)

    def _noise_generator(self, x: t.Tensor):
        if self._iter is None:
            self._iter = iter(
                DataLoader(self.dataloader, num_workers=0, batch_size=None)
            )

        noise_dict = next(self._iter, None)

        if noise_dict is None:
            raise ValueError("ran out of noise samples")

        noise_tensor, noise_sr = noise_dict["wav"]
        inp_source_length = x.shape[1]

        # repeat noise until it can fit the whole audio file
        while noise_tensor.shape[1] < inp_source_length:
            noise_tensor = noise_tensor.repeat(1, 2)

        return noise_tensor[:, 0:inp_source_length]

    def _random_snr(self):
        return np.random.choice(self.snr_choices)

    def _get_chain(self) -> wavaugment.EffectChain:
        raise NotImplementedError()

    def process(self, x: t.Tensor) -> [t.Tensor]:
        noise = self._noise_generator(x)

        chain = wavaugment.EffectChain().additive_noise(
            lambda: noise, snr=self._random_snr()
        )

        return chain.apply(
            x,
            src_info={"rate": self.sample_rate},
            target_info={"channels": 1, "rate": self.sample_rate},
        )


################################################################################
# Randomly add reverb to the audio based on a certain room size


class ReverbAugment(WavAugmentWrapper):
    def __init__(
        self,
        sample_rate: int,
        reverberance_min: int = 50,
        reverberance_max: int = 50,
        damping_min: int = 50,
        damping_max: int = 50,
        room_scale_min: int = 0,
        room_scale_max: int = 100,
    ):
        super().__init__(sample_rate, "add_reverb")

        self.reverberance_min = reverberance_min
        self.reverberance_max = reverberance_max
        self.damping_min = damping_min
        self.damping_max = damping_max
        self.room_scale_min = room_scale_min
        self.room_scale_max = room_scale_max

        self._chain = (
            wavaugment.EffectChain()
            .reverb(
                self._get_random_reverberance(),
                self._get_random_damping(),
                self._get_random_room_scale(),
            )
            .channels()
        )

    def _get_random_reverberance(self):
        return np.random.randint(self.reverberance_min, self.reverberance_max + 1)

    def _get_random_damping(self):
        return np.random.randint(self.damping_min, self.damping_max + 1)

    def _get_random_room_scale(self):
        return np.random.randint(self.room_scale_min, self.room_scale_max + 1)

    def _get_chain(self) -> wavaugment.EffectChain:
        return self._chain
