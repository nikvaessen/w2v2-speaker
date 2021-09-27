################################################################################
#
# Implement the wav2vec + x-vector network head for speaker recognition as a
# SpeakerRecognitionModule.
#
# Author(s): Nik Vaessen
################################################################################

import pathlib

from dataclasses import dataclass
from typing import List, Optional, Callable

import torch
import torch as t
import torch.nn as nn
import speechbrain as sb

from omegaconf import DictConfig

from src.evaluation.speaker.speaker_recognition_evaluator import (
    EvaluationPair,
    SpeakerRecognitionEvaluator,
)
from src.layers.pooling import MeanStdStatPool1D, MeanStatPool1D
from src.optim.loss.aam_softmax import AngularAdditiveMarginSoftMaxLoss
from src.models.wav2vec import Wav2VecWrapperModule
from src.lightning_modules.speaker.speaker_recognition_module import (
    SpeakerRecognitionLightningModule,
)

################################################################################
# Implementation of wav2vec with x-vector network head


@dataclass
class Wav2vecFCModuleConfig:
    # settings for wav2vec architecture
    wav2vec_model_path: str
    use_aggregation_layers: bool
    reset_weights: bool

    # settings related to training wav2vec
    wav2vec_initially_frozen: bool
    num_frozen_steps: Optional[int]

    # settings for fc head
    hidden_fc_layers_out: List[int]
    embedding_layer_idx: int
    stat_pooling_type: str


class Wav2vecFCModule(SpeakerRecognitionLightningModule):
    def __init__(
        self,
        hyperparameters_to_save: DictConfig,
        cfg: Wav2vecFCModuleConfig,
        num_speakers: int,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        validation_pairs: List[EvaluationPair],
        test_pairs: List[EvaluationPair],
        evaluator: SpeakerRecognitionEvaluator,
    ):
        self.cfg = cfg

        super().__init__(
            hyperparameter_config=hyperparameters_to_save,
            num_speakers=num_speakers,
            embedding_size=self._determine_embedding_size(),
            loss_fn_constructor=loss_fn_constructor,
            validation_pairs=validation_pairs,
            test_pairs=test_pairs,
            evaluator=evaluator,
        )

        # create base wav2vec model
        self.wav2vec = Wav2VecWrapperModule(
            wav2vec_model_path=pathlib.Path(cfg.wav2vec_model_path),
            wav2vec_aggregation=cfg.use_aggregation_layers,
            reset_weights=self.cfg.reset_weights,
        )
        self._is_wav2vec_frozen = False

        # create stat_pool layer
        if self.cfg.stat_pooling_type == "mean":
            self.stat_pooling = MeanStatPool1D(dim_to_reduce=1)
        elif self.cfg.stat_pooling_type == "mean+std":
            self.stat_pooling = MeanStdStatPool1D(dim_to_reduce=1)
        else:
            raise ValueError(
                f"unknown value {cfg.stat_pooling_type=}, should be one of "
                f"['mean', 'mean+std']"
            )
        self.stat_pool_dimension = self._determine_stat_pool_embedding_size()

        # create fc layers

        self.fc_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        in_features=self.stat_pool_dimension
                        if idx == 0
                        else cfg.hidden_fc_layers_out[idx - 1],
                        out_features=num_out,
                    ),
                    nn.ReLU(),
                )
                for idx, num_out in enumerate(cfg.hidden_fc_layers_out)
            ]
        )
        self.fc_list.append(
            nn.Sequential(
                nn.Linear(
                    in_features=self.stat_pool_dimension
                    if len(cfg.hidden_fc_layers_out) == 0
                    else cfg.hidden_fc_layers_out[-1],
                    out_features=self.num_speakers,
                ),
                sb.nnet.activations.Softmax(apply_log=True),
            )
        )

        if isinstance(self.loss_fn, AngularAdditiveMarginSoftMaxLoss):
            raise ValueError("wav2vec_fc does not support aam softmax")

        self.steps = 0

    def _determine_embedding_size(self):
        if self.cfg.embedding_layer_idx < 0:
            return self._determine_stat_pool_embedding_size()
        elif 0 <= self.cfg.embedding_layer_idx < len(self.cfg.hidden_fc_layers_out):
            return self.cfg.hidden_fc_layers_out[self.cfg.embedding_layer_idx]
        elif self.cfg.embedding_layer_idx == len(self.cfg.hidden_fc_layers_out):
            return self.num_speakers
        else:
            raise ValueError("could not determine size of speaker embeddings")

    def _determine_stat_pool_embedding_size(self):
        if self.cfg.stat_pooling_type == "mean":
            return self.wav2vec.num_features  # output of wav2vec embedding size
        elif self.cfg.stat_pooling_type == "mean+std":
            return self.wav2vec.num_features * 2  # output of wav2vec embedding size
        else:
            raise ValueError(f"unknown value for {self.cfg.stat_pooling_type=}")

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

    def _fc_head_ops_pre_spk_embedding(self, wav2vec_embedding: t.tensor):
        # first apply stat pooling
        pooled_embedding = self.stat_pooling(wav2vec_embedding)

        if self.cfg.embedding_layer_idx < 0:
            return pooled_embedding

        # loop over fc layers until we have reached the index
        # from which to select the speaker embedding
        x = pooled_embedding

        for idx, fc_layer in enumerate(self.fc_list):
            x = fc_layer(x)

            if self.cfg.embedding_layer_idx == idx:
                break

        return x

    def _fc_head_ops_post_spk_embedding(self, embedding_tensor: t.Tensor):
        # loop over fc layers which were not yet used in
        # `_fc_head_ops_pre_spk_embedding`
        x = embedding_tensor

        for idx, fc_layer in enumerate(self.fc_list):
            if idx <= self.cfg.embedding_layer_idx:
                continue

            x = fc_layer(x)

        prediction_tensor = x

        return prediction_tensor

    def compute_speaker_embedding(self, input_tensor: t.Tensor) -> t.Tensor:
        # transform input
        # (of shape [BS, 1, NUM_AUDIO_SAMPLES] or [1, NUM_AUDIO_SAMPLES])
        # to the required [BS, NUM_AUDIO_SAMPLES]
        if len(input_tensor.shape) == 3 and input_tensor.shape[1] == 1:
            input_tensor = torch.squeeze(input_tensor)
        if len(input_tensor.shape) == 1:
            input_tensor = torch.stack([input_tensor])

        # first compute the wav2vec embeddings: will be shape [BS, 512, NUM_WINDOWS]
        wav2vec_embeddings = self.wav2vec(input_tensor)

        # then we need to transpose to the expected input shape of
        # [BS, NUM_WINDOWS, NUM_FEATURES] by the rest of the network
        wav2vec_embeddings = t.transpose(wav2vec_embeddings, 2, 1)

        # we end with all the operations to get to the speaker embeddings
        return self._fc_head_ops_pre_spk_embedding(wav2vec_embeddings)

    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        # we apply all operations we need to apply on the speaker
        # embedding to get to the classification prediction
        prediction = self._fc_head_ops_post_spk_embedding(embedding_tensor)

        return prediction.squeeze()
