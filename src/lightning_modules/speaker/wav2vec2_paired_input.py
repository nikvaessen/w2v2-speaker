################################################################################
#
# Implement wav2vec2 with 2 audio samples in forward pass
# and logistical regression prediction head for equal/not equal
#
# Author(s): Nik Vaessen
################################################################################

from dataclasses import dataclass
from typing import Optional, Callable

import torch as t
import torch.nn as nn

from omegaconf import DictConfig

from src.lightning_modules.speaker.paired_speaker_recognition_module import (
    PairedSpeakerRecognitionLightningModule,
)
from src.models.wav2vec2 import Wav2Vec2WrapperModule, Wav2Vec2RegularisationConfig


################################################################################
# Implementation of wav2vec2 + paired input for speaker equality prediction


@dataclass
class Wav2vec2PairedSpeakerModuleConfig:
    # settings for wav2vec architecture
    wav2vec_hunggingface_id: str
    reset_weights: bool

    # settings related to training wav2vec2
    wav2vec_initially_frozen: bool
    num_frozen_steps: Optional[int]
    completely_freeze_feature_extractor: bool
    completely_freeze_feature_projector: bool

    # constant values to fill initial tokens
    cls_token_constant = 1
    sep_token_constant = -1

    # probability of regularization techniques during training
    # dropout
    activation_dropout: float  # in feed-forward module of transformer layer
    attention_dropout: float  # in attention module of transformer layer
    feat_proj_dropout: float  # in feature projection module
    hidden_dropout: float  # between residual connections in transformer layer

    # layer skip in transformer
    layerdrop: float

    # specaugment
    # feature
    mask_feature_length: int
    mask_feature_prob: float
    # time
    mask_time_length: int
    mask_time_prob: float

    # settings for dropout on final output embedding
    final_channel_mask_prob: float
    final_channel_mask_width: int


class Wav2vec2PairedSpeakerModule(PairedSpeakerRecognitionLightningModule):
    def __init__(
        self,
        hyperparameters_to_save: DictConfig,
        cfg: Wav2vec2PairedSpeakerModuleConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
    ):
        self.cfg = cfg

        # initialize as super class
        super().__init__(
            hyperparameter_config=hyperparameters_to_save,
            loss_fn_constructor=loss_fn_constructor,
        )

        # create base wav2vec model
        self.wav2vec = Wav2Vec2WrapperModule(
            wav2vec2_huggingface_id=cfg.wav2vec_hunggingface_id,
            reset_weights=self.cfg.reset_weights,
            reg_cfg=Wav2Vec2RegularisationConfig(
                gradient_checkpointing=False,
                activation_dropout=self.cfg.activation_dropout,
                attention_dropout=self.cfg.attention_dropout,
                feat_proj_dropout=self.cfg.feat_proj_dropout,
                hidden_dropout=self.cfg.hidden_dropout,
                layerdrop=self.cfg.layerdrop,
                mask_feature_length=self.cfg.mask_feature_length,
                mask_feature_prob=self.cfg.mask_feature_prob,
                mask_time_length=self.cfg.mask_time_length,
                mask_time_prob=self.cfg.mask_time_prob,
            ),
        )
        self._is_wav2vec_frozen = False

        # prediction head
        self.linear = nn.Linear(
            in_features=self._get_wav2vec2_embedding_size(), out_features=1
        )

        self.steps = 0

    def _get_wav2vec2_embedding_size(self):
        if "base" in self.cfg.wav2vec_hunggingface_id:
            return 768
        elif "large" in self.cfg.wav2vec_hunggingface_id:
            return 1024
        else:
            raise ValueError("unknown wav2ec2 embedding size")

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

        return t.rand(size=shape), t.rand(size=shape)

    def on_train_start(self) -> None:
        self.steps = 0

        if self.cfg.wav2vec_initially_frozen:
            self.wav2vec.freeze()
            self._is_wav2vec_frozen = True

        if self.cfg.completely_freeze_feature_extractor:
            self.wav2vec.model.feature_extractor.requires_grad_(False)

        if self.cfg.completely_freeze_feature_projector:
            self.wav2vec.model.feature_projection.requires_grad_(False)

    def on_after_backward(self) -> None:
        self.steps += 1

        if (
            self._is_wav2vec_frozen
            and self.cfg.num_frozen_steps is not None
            and self.steps >= self.cfg.num_frozen_steps
        ):
            self.wav2vec.unfreeze()
            self._is_wav2vec_frozen = False

            if self.cfg.completely_freeze_feature_extractor:
                self.wav2vec.model.feature_extractor.requires_grad_(False)

            if self.cfg.completely_freeze_feature_projector:
                self.wav2vec.model.feature_projection.requires_grad_(False)

    def compute_speaker_equality(
        self, wav_tensor: t.Tensor, other_wav_tensor
    ) -> t.Tensor:
        # assert equal batch sizes and devices
        assert wav_tensor.shape[0] == wav_tensor.shape[0]
        assert wav_tensor.device == wav_tensor.device

        # first do feature extraction
        features_1 = self.wav2vec.model.feature_extractor(wav_tensor)
        features_2 = self.wav2vec.model.feature_extractor(other_wav_tensor)

        features_1 = features_1.transpose(1, 2)
        features_2 = features_2.transpose(1, 2)

        # project 516 features to 768/1024 features (for base/large model)
        features_1, _ = self.wav2vec.model.feature_projection(features_1)
        features_2, _ = self.wav2vec.model.feature_projection(features_2)

        # concatenate to sequence before feeding to encoder
        cls_token = (
            t.ones((wav_tensor.shape[0], 1, 768), device=wav_tensor.device)
            * self.cfg.cls_token_constant
        )
        sep_token = (
            t.ones((wav_tensor.shape[0], 1, 768), device=wav_tensor.device)
            * self.cfg.sep_token_constant
        )
        end_token = (
            t.ones((wav_tensor.shape[0], 1, 768), device=wav_tensor.device)
            * self.cfg.sep_token_constant
        )

        sequence = t.cat(
            [cls_token, features_1, sep_token, features_2, end_token], dim=1
        )

        # compute tokens
        encoder_output = self.wav2vec.model.encoder(sequence)
        wav2vec2_tokens = encoder_output.last_hidden_state
        cls_token = wav2vec2_tokens[:, 0, :]

        # predict equality
        equality_score = self.linear(cls_token)

        return equality_score
