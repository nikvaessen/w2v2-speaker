################################################################################
#
# Implement wav2vec2 as speech recognition module
#
# Author(s): Nik Vaessen
################################################################################

import math

from typing import Optional, Callable, List, Dict, Tuple

import torch as t
import torch.nn as nn

from dataclasses import dataclass
from omegaconf import DictConfig
from transformers.models.wav2vec2 import Wav2Vec2ForCTC

from src.layers.embedding_masking import EmbeddingMasker
from src.lightning_modules.speech.speech_recognition_module import (
    SpeechRecognitionLightningModule,
)

################################################################################
# config
from src.models.wav2vec2 import Wav2Vec2WrapperModule
from src.tokenizer.base import BaseTokenizer


@dataclass
class Wav2vec2FcLetterRecognizerConfig:
    # pretrained weights of wav2vec model
    wav2vec_hunggingface_id: str

    # whether to use reset the pretrained weights
    # and start from a fresh initialization
    reset_weights: bool

    # initially freeze wav2vec model
    wav2vec_initially_frozen: bool

    # whether to freeze the feature encoder part
    # of the network for the whole training run
    completely_freeze_feature_extractor: bool

    # number of steps before the wav2vec model is unfrozen
    # (if initially frozen at all)
    # if set to null, wav2vec will never be unfrozen
    num_frozen_steps: int

    # mask (dropout of embedding tensor) settings
    timestep_mask_prob: float
    timestep_mask_width: int
    channel_mask_prob: float
    channel_mask_width: int

    # language recognition head pretrained weights
    speech_head_huggingface_id: Optional[str] = None


################################################################################
# Lightning module


class SpeechRecognitionHead(nn.Module):
    def __init__(self, huggingface_wav2vec2_ctc_id: Optional[str]):
        super().__init__()

        if huggingface_wav2vec2_ctc_id is not None:
            tmp_init = Wav2Vec2ForCTC.from_pretrained(huggingface_wav2vec2_ctc_id)
        else:
            tmp_init = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")

        self.dropout = tmp_init.dropout
        self.lm_head = tmp_init.lm_head

        del tmp_init

    def forward(self, wav2vec2_embedding: t.Tensor):
        # apply dropout on embeddings
        wav2vec2_embedding = self.dropout(wav2vec2_embedding)

        # use linear head to predict letter for each embedding in the sequence
        letter_predictions = self.lm_head(wav2vec2_embedding)

        return letter_predictions


class Wav2vec2FcLetterRecognizer(SpeechRecognitionLightningModule):
    def __init__(
        self,
        cfg: Wav2vec2FcLetterRecognizerConfig,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        tokenizer: BaseTokenizer,
    ):
        self.cfg = cfg

        super().__init__(hyperparameter_config, loss_fn_constructor, tokenizer)

        # state variables for freezing network
        self.steps = 0
        self._is_wav2vec_frozen = False

        # load network
        self.wav2vec = Wav2Vec2WrapperModule(
            wav2vec2_huggingface_id=cfg.wav2vec_hunggingface_id,
            reset_weights=self.cfg.reset_weights,
        )

        self.speech_recognition_head = SpeechRecognitionHead(
            cfg.wav2vec_hunggingface_id
        )

        self.embedding_masker = EmbeddingMasker(
            timestep_mask_prob=self.cfg.timestep_mask_prob,
            timestep_mask_width=self.cfg.timestep_mask_width,
            channel_mask_prob=self.cfg.channel_mask_prob,
            channel_mask_width=self.cfg.channel_mask_width,
            time_dim=1,
            embedding_dim=2,
        )

    def compute_speech_embedding(
        self, input_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        # transform input
        # (of shape [BS, 1, NUM_AUDIO_SAMPLES] or [1, NUM_AUDIO_SAMPLES])
        # to the required [BS, NUM_AUDIO_SAMPLES]
        if len(input_tensor.shape) == 3 and input_tensor.shape[1] == 1:
            input_tensor = t.squeeze(input_tensor)
        if len(input_tensor.shape) == 1:
            input_tensor = t.stack([input_tensor])

        # first compute the wav2vec embeddings: will be shape [BS, 768, NUM_WINDOWS]
        wav2vec_embeddings = self.wav2vec(input_tensor)

        # then we need to transpose to the expected input shape of
        # [BS, NUM_WINDOWS, NUM_FEATURES] by the rest of the network
        wav2vec_embeddings = t.transpose(wav2vec_embeddings, 2, 1)

        # apply masking (dropout)
        wav2vec_embeddings = self.embedding_masker(wav2vec_embeddings)

        # calculate the new lengths
        lengths = [math.floor((num_frames - 80) / 320) for num_frames in lengths]

        # we end with all the operations to get to the speaker embeddings
        return wav2vec_embeddings, lengths

    def compute_vocabulary_prediction(
        self, embedding_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        # lengths is not modified by head
        letter_prediction = self.speech_recognition_head(embedding_tensor)

        return letter_prediction, lengths

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

        return t.rand(size=shape), [16000]

    def on_train_start(self) -> None:
        self.steps = 0

        if self.cfg.wav2vec_initially_frozen:
            self.wav2vec.freeze()
            self._is_wav2vec_frozen = True

        if self.cfg.completely_freeze_feature_extractor:
            self.wav2vec.model.feature_extractor.requires_grad_(False)

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
