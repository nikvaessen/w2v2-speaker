################################################################################
#
# Implement the x-vector network for speaker recognition as a
# SpeakerRecognitionModule.
#
# Author(s): Nik Vaessen
################################################################################

import pathlib

from dataclasses import dataclass, field
from typing import List, Optional, Callable

import torch
import torch as t

from omegaconf import DictConfig
from speechbrain.lobes.models.Xvector import Xvector, Classifier

from src.evaluation.speaker.speaker_recognition_evaluator import (
    SpeakerRecognitionEvaluator,
    EvaluationPair,
)
from src.optim.loss.aam_softmax import AngularAdditiveMarginSoftMaxLoss
from src.models.wav2vec import Wav2VecWrapperModule
from src.lightning_modules.speaker.speaker_recognition_module import (
    SpeakerRecognitionLightningModule,
)

################################################################################
# Implementation of wav2vec with x-vector network head


@dataclass
class Wav2vecXVectorModuleConfig:
    # settings for wav2vec architecture
    wav2vec_model_path: str
    use_aggregation_layers: bool

    # settings related to training wav2vec
    wav2vec_initially_frozen: bool
    num_frozen_steps: Optional[int]

    # settings for xvector network
    tdnn_blocks: int = 5
    tdnn_channels: List[int] = field(default_factory=[512, 512, 512, 512, 1500])
    tdnn_kernel_sizes: List[int] = field(default_factory=[5, 3, 3, 1, 1])
    tdnn_dilations: List[int] = field(default_factory=[1, 2, 3, 1, 1])
    lin_neurons: int = 512
    in_channels: int = 40


class Wav2vecXVectorModule(SpeakerRecognitionLightningModule):
    def __init__(
        self,
        hyperparameters_to_save: DictConfig,
        cfg: Wav2vecXVectorModuleConfig,
        num_speakers: int,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        validation_pairs: List[EvaluationPair],
        test_pairs: List[EvaluationPair],
        evaluator: SpeakerRecognitionEvaluator,
    ):
        super().__init__(
            hyperparameter_config=hyperparameters_to_save,
            num_speakers=num_speakers,
            embedding_size=self._determine_embedding_size(),
            loss_fn_constructor=loss_fn_constructor,
            validation_pairs=validation_pairs,
            test_pairs=test_pairs,
            evaluator=evaluator,
        )

        self.cfg = cfg

        self.wav2vec = Wav2VecWrapperModule(
            wav2vec_model_path=pathlib.Path(cfg.wav2vec_model_path),
            wav2vec_aggregation=cfg.use_aggregation_layers,
        )
        self._is_wav2vec_frozen = False

        self.feature_extractor = Xvector(
            tdnn_blocks=cfg.tdnn_blocks,
            tdnn_channels=cfg.tdnn_channels,
            tdnn_kernel_sizes=cfg.tdnn_kernel_sizes,
            tdnn_dilations=cfg.tdnn_dilations,
            lin_neurons=cfg.lin_neurons,
            in_channels=cfg.in_channels,
        )
        self.classifier = Classifier(
            (None, None, self.embedding_size), out_neurons=self.num_speakers
        )

        if isinstance(self.loss_fn, AngularAdditiveMarginSoftMaxLoss):
            raise ValueError("wav2vec-xvector does not support aam softmax")

        self.steps = 0

    def generate_example_input(
        self, include_batch_dimension: bool, batch_size: Optional[int] = None
    ):
        if include_batch_dimension:
            # [BATCH_SIZE, NUMBER_OF_AUDIO_SAMPLES]
            # the `16000` varies depending on length of audio file
            # (1 second in this case)
            shape = [batch_size, 16000]
        else:
            # [BATCH_SIZE, NUMBER_OF_AUDIO_SAMPLES]
            # the `16000` varies depending on length of audio file
            # (1 second in this case)
            shape = [
                16000,
            ]

        return t.rand(size=shape)

    def on_train_start(self) -> None:
        self.steps = 0

        if self.cfg.wav2vec_initially_frozen:
            self.wav2vec.freeze()
            self._is_wav2vec_frozen = True

    def on_after_backward(self) -> None:
        self.steps += 1

        if (
            self._is_wav2vec_frozen
            and self.cfg.num_frozen_steps is not None
            and self.steps >= self.cfg.num_frozen_steps
        ):
            self.wav2vec.unfreeze()
            self._is_wav2vec_frozen = False

    def compute_speaker_embedding(self, input_tensor: t.Tensor) -> t.Tensor:
        # transform input
        # (of shape [BS, 1, NUM_AUDIO_SAMPLES] or [1, NUM_AUDIO_SAMPLES])
        # to the required [BS, NUM_AUDIO_SAMPLES]
        if len(input_tensor.shape) == 3 and input_tensor.shape[1] == 1:
            input_tensor = torch.squeeze(input_tensor)
        if len(input_tensor.shape) == 1:
            input_tensor = torch.stack([input_tensor])

        # first compute the wav2vec embeddings: will be shape [BS, 512, NUM_WINDOWS]
        wav2vec_embedding = self.wav2vec(input_tensor)

        # then we need to transpose to the expected input shape of
        # [BS, NUM_WINDOWS, NUM_FEATURES] of the xvector network
        wav2vec_embedding = t.transpose(wav2vec_embedding, 2, 1)

        # now we can process input with xvector network
        x = wav2vec_embedding

        for layer in self.feature_extractor.blocks:
            x = layer(x)

        if len(x.shape) == 3:
            x = x.squeeze()

        if len(x.shape) == 1:
            x = torch.stack([x])

        return x

    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        # classifier expect shape [BATCH_SIZE, 1, EMBEDDING_SIZE]
        embedding_tensor = embedding_tensor[:, None, :]
        prediction = self.classifier(embedding_tensor)

        return prediction.squeeze()
