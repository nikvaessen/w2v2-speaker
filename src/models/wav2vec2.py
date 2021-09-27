################################################################################
#
# Provide embeddings from raw audio with the wav2vec2 model from huggingface.
#
# Author(s): Nik Vaessen
################################################################################

import pathlib
from dataclasses import dataclass
from typing import Optional

import torch as t
import pytorch_lightning as pl

from transformers.models.wav2vec2 import Wav2Vec2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2BaseModelOutput

from src.util import reset_model


################################################################################
# loading wav2vec with fairseq


def load_base_wav2vec2_model(
    huggingface_id: str,
    reg_cfg: Optional["Wav2Vec2RegularisationConfig"] = None,
    device: t.cuda.Device = t.device("cpu"),
) -> Wav2Vec2Model:
    """
    Load the wav2vec2 model.

    :param huggingface_id: huggingface identifier of pretrained model
    :param device: the device on which the model should be loaded
    :return: the wav2vec2 model on the specified device
    """
    if reg_cfg is not None:
        model = Wav2Vec2Model.from_pretrained(
            huggingface_id,
            # overwrite values in config
            gradient_checkpointing=reg_cfg.gradient_checkpointing,
            activation_dropout=reg_cfg.activation_dropout,
            attention_dropout=reg_cfg.attention_dropout,
            feat_proj_dropout=reg_cfg.feat_proj_dropout,
            hidden_dropout=reg_cfg.hidden_dropout,
            layerdrop=reg_cfg.layerdrop,
            mask_feature_length=reg_cfg.mask_feature_length,
            mask_feature_prob=reg_cfg.mask_feature_prob,
            mask_time_length=reg_cfg.mask_time_length,
            mask_time_prob=reg_cfg.mask_time_prob,
        )
    else:
        model = Wav2Vec2Model.from_pretrained(huggingface_id)

    return model.to(device)


################################################################################
# computation of embedding


def wav2vec2_embed_raw_audio(input_tensor: t.Tensor, model: Wav2Vec2Model) -> t.Tensor:
    """
    Calculate a [1, 768, num_frames] embedding of a given [1, num_samples] audio file
    by using the Wav2Vec2 model.

    :param input_tensor: a raw audio input (between -1 and 1) with a sampling rate of 16000 Hz
    :param model: the wav2vec model
    :return: The embedding with shape [1, 768, num_frames], where num_frames < num_samples.
    """
    output: Wav2Vec2BaseModelOutput = model(input_tensor)

    features = output.last_hidden_state
    features = features.transpose(1, 2)

    return features


################################################################################
# Wrap functionality in a LightningModule


@dataclass
class Wav2Vec2RegularisationConfig:
    gradient_checkpointing: bool = False
    activation_dropout: float = 0.0
    attention_dropout: float = 0.1
    feat_proj_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layerdrop: float = 0.05
    mask_feature_length: int = 10
    mask_feature_prob: float = 0.0
    mask_time_length: int = 10
    mask_time_prob: float = 0.05


class Wav2Vec2WrapperModule(pl.LightningModule):
    def __init__(
        self,
        wav2vec2_huggingface_id: str,
        reset_weights: bool,
        reg_cfg: Optional[Wav2Vec2RegularisationConfig] = None,
        insert_clc_token: bool = False,
        cls_token_constant: float = 1,
    ):
        super().__init__()

        self.model = load_base_wav2vec2_model(wav2vec2_huggingface_id, reg_cfg)
        self.insert_cls_token = insert_clc_token
        self.cls_token_constant = cls_token_constant

        if "base" in wav2vec2_huggingface_id:
            self.num_features = 768
        elif "large" in wav2vec2_huggingface_id:
            self.num_features = 1024
        else:
            raise ValueError("cannot determine num features")

        if reset_weights:
            reset_model(self.model)

    @property
    def num_embedding_features(self):
        return self.num_features

    def forward(self, wav_input: t.Tensor):
        # wav_input has shape [BATCH_SIZE, NUM_SAMPLES]
        if self.insert_cls_token:
            features = self.model.feature_extractor(wav_input)
            features = features.transpose(1, 2)
            features, _ = self.model.feature_projection(features)
            cls_token = (
                t.ones((wav_input.shape[0], 1, 768), device=wav_input.device)
                * self.cls_token_constant
            )
            sequence = t.cat([cls_token, features], dim=1)

            encoder_output = self.model.encoder(sequence)
            embedding = encoder_output.last_hidden_state
            embedding = embedding.transpose(1, 2)

        else:
            embedding = wav2vec2_embed_raw_audio(wav_input, self.model)

        # return an embedding with shape [BATCH_SIZE, NUM_FEATURES, NUM_FRAMES]
        return embedding


class Wav2vecLiteWrapperModule(pl.LightningModule):
    num_features = 512

    def __init__(self, wav2vec2_huggingface_id: str, reset_weights: bool):
        super().__init__()

        self.model = load_base_wav2vec2_model(wav2vec2_huggingface_id)

        if reset_weights:
            reset_model(self.model)

    @property
    def num_embedding_features(self):
        return self.num_features

    def forward(self, wav_input: t.Tensor):
        # wav_input has shape [BATCH_SIZE, NUM_SAMPLES]
        embedding = self.model.feature_extractor(wav_input)

        # return an embedding with shape [BATCH_SIZE, NUM_FEATURES, NUM_FRAMES]
        return embedding
