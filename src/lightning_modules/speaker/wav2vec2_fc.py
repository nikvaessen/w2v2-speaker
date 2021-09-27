################################################################################
#
# Implement the wav2vec2 + fc network head for speaker recognition as a
# SpeakerRecognitionModule.
#
# Author(s): Nik Vaessen
################################################################################

from dataclasses import dataclass
from re import S
from src.optim.loss.ctc_loss import CtcLoss
from typing import List, Optional, Callable

import torch
import torch as t
import torch.nn as nn

from omegaconf import DictConfig

from src.evaluation.speaker.speaker_recognition_evaluator import (
    EvaluationPair,
    SpeakerRecognitionEvaluator,
)
from src.layers.embedding_masking import EmbeddingMasker
from src.layers.pooling import (
    MeanStdStatPool1D,
    MeanStatPool1D,
    AttentiveStatPool1D,
    QuantilePool1D,
    IndexPool1D,
    NoPooling,
    MaxPool1D,
)
from src.optim.loss.aam_softmax import AngularAdditiveMarginSoftMaxLoss
from src.models.wav2vec2 import (
    Wav2Vec2WrapperModule,
    Wav2vecLiteWrapperModule,
    Wav2Vec2RegularisationConfig,
)
from src.lightning_modules.speaker.speaker_recognition_module import (
    SpeakerRecognitionLightningModule,
)

################################################################################
# Implementation of wav2vec with x-vector network head


@dataclass
class Wav2vec2FCModuleConfig:
    # settings for wav2vec architecture
    wav2vec_hunggingface_id: str
    reset_weights: bool

    # settings related to wav2vec2 architecture
    wav2vec_feature_encoder_only: bool

    # settings related to training wav2vec2
    wav2vec_initially_frozen: bool
    num_frozen_steps: Optional[int]
    completely_freeze_feature_extractor: bool

    # settings for fc head
    hidden_fc_layers_out: List[int]
    embedding_layer_idx: int
    stat_pooling_type: str
    test_stat_pooling_type: str

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

    # optional explicit overwrite of embedding size (e.g if you
    # need to load finetuned weights but want to experiment with another
    # pooling type in the evaluation)
    explicit_stat_pool_embedding_size: Optional[int]
    explicit_num_speakers: Optional[int]

    # test with ensemble of all transformer layers
    use_transformers_as_ensembles: bool = False
    num_ensembles: int = 12


class Wav2vec2FCModule(SpeakerRecognitionLightningModule):
    def __init__(
        self,
        hyperparameters_to_save: DictConfig,
        cfg: Wav2vec2FCModuleConfig,
        num_speakers: int,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        validation_pairs: List[EvaluationPair],
        test_pairs: List[EvaluationPair],
        evaluator: SpeakerRecognitionEvaluator,
    ):
        self.cfg = cfg

        if cfg.wav2vec_feature_encoder_only:
            self.wav2vec_wrapper_class = Wav2vecLiteWrapperModule
        else:
            self.wav2vec_wrapper_class = Wav2Vec2WrapperModule

        if (
            self.cfg.completely_freeze_feature_extractor
            and self.cfg.wav2vec_feature_encoder_only
        ):
            raise ValueError(
                "can not freeze the whole network! "
                "Either `completely_freeze_feature_extractor` or "
                "`wav2vec_feature_encoder_only` need to be set to False"
            )

        # initialize as super class
        super().__init__(
            hyperparameter_config=hyperparameters_to_save,
            num_speakers=num_speakers,
            embedding_size=self._determine_embedding_size(),
            loss_fn_constructor=loss_fn_constructor,
            validation_pairs=validation_pairs,
            test_pairs=test_pairs,
            evaluator=evaluator,
            embeddings_are_pooled=self.cfg.stat_pooling_type != "none",
        )

        # create base wav2vec model
        self.wav2vec = self.wav2vec_wrapper_class(
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
            insert_clc_token=self.cfg.stat_pooling_type == "first+cls",
        )
        self._is_wav2vec_frozen = False

        # create masker
        self.embedding_masker = EmbeddingMasker(
            timestep_mask_prob=0,
            timestep_mask_width=1,
            channel_mask_prob=self.cfg.final_channel_mask_prob,
            channel_mask_width=self.cfg.final_channel_mask_width,
            time_dim=2,
            embedding_dim=1,
        )

        # create stat_pool layer
        self.stat_pooling = self._determine_pooling_layer(
            self.cfg.stat_pooling_type, only_at_test_time=False
        )
        self.stat_pool_dimension = self._determine_stat_pool_embedding_size()

        if self.cfg.test_stat_pooling_type != self.cfg.stat_pooling_type:
            self.test_stat_pooling = self._determine_pooling_layer(
                self.cfg.test_stat_pooling_type, only_at_test_time=True
            )
        else:
            self.test_stat_pooling = self.stat_pooling

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
                    out_features=self.cfg.explicit_num_speakers
                    if self.cfg.explicit_num_speakers
                    else self.num_speakers,
                ),
            )
        )

        if isinstance(self.loss_fn, AngularAdditiveMarginSoftMaxLoss):
            # remove the last layer and replace it with the AAM softmax loss
            del self.fc_list[-1]

            # recreate AAM loss with correct input/output features
            self.loss_fn = AngularAdditiveMarginSoftMaxLoss(
                input_features=self.stat_pool_dimension,
                output_features=self.cfg.explicit_num_speakers
                if self.cfg.explicit_num_speakers is not None
                else self.num_speakers,
                margin=self.loss_fn.margin,
                scale=self.loss_fn.scale,
            )

        if isinstance(self.loss_fn, CtcLoss):
            # set bias of final fc layer to high prior for blank
            final_out = self.fc_list[-1][0]

            new_bias = t.zeros(final_out.bias.shape)
            new_bias[0] = 100

            final_out.bias = nn.Parameter(new_bias)

        self.steps = 0
        self.test_with_ensemble = self.cfg.use_transformers_as_ensembles

    def _determine_pooling_layer(self, stat_pooling_type: str, only_at_test_time: bool):
        if stat_pooling_type == "mean":
            stat_pooling = MeanStatPool1D(dim_to_reduce=1)
        elif stat_pooling_type == "mean+std":
            stat_pooling = MeanStdStatPool1D(dim_to_reduce=1)
        elif stat_pooling_type == "attentive":
            if only_at_test_time:
                raise ValueError("attention can not be learned at test time")
            stat_pooling = AttentiveStatPool1D(
                dim_to_reduce=1, embedding_size=self.wav2vec.num_features
            )
        elif stat_pooling_type == "quantile":
            stat_pooling = QuantilePool1D(dim_to_reduce=1)
        elif stat_pooling_type in [
            "first",
            "first+cls",
            "last",
            "middle",
            "random",
        ]:
            stat_pooling = IndexPool1D(
                selection_method=self.cfg.stat_pooling_type, dim_to_reduce=1
            )
        elif stat_pooling_type == "max":
            stat_pooling = MaxPool1D(dim_to_reduce=1)
        elif stat_pooling_type.lower() == "none":
            stat_pooling = NoPooling()
        else:
            raise ValueError(
                f"unknown value {stat_pooling_type=}, should be one of "
                f"['mean', 'mean+std', 'attentive', 'quantile', 'max',"
                f" 'first', 'last', 'middle', 'random', 'none']"
            )

        return stat_pooling

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
        if self.cfg.explicit_stat_pool_embedding_size is not None:
            return self.cfg.explicit_stat_pool_embedding_size

        if self.cfg.stat_pooling_type.lower() in [
            "mean",
            "first",
            "first+cls",
            "last",
            "middle",
            "random",
            "max",
            "none",
        ]:
            return (
                self._get_wav2vec2_embedding_size()
            )  # output of wav2vec embedding size
        elif (
            self.cfg.stat_pooling_type == "mean+std"
            or self.cfg.stat_pooling_type == "attentive"
        ):
            return (
                self._get_wav2vec2_embedding_size() * 2
            )  # output of wav2vec embedding size
        elif self.cfg.stat_pooling_type == "quantile":
            return self._get_wav2vec2_embedding_size() * 5
        else:
            raise ValueError(f"unknown value for {self.cfg.stat_pooling_type=}")

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

        return t.rand(size=shape)

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

    def _fc_head_ops_pre_spk_embedding(self, wav2vec_embedding: t.tensor):
        # first apply stat pooling
        if self.training:
            pooled_embedding = self.stat_pooling(wav2vec_embedding)
        else:
            pooled_embedding = self.test_stat_pooling(wav2vec_embedding)

        # assert equal batch size, correct dimension and shape of len 2
        if not isinstance(self.stat_pooling, NoPooling):
            assert pooled_embedding.shape[1] == self.stat_pool_dimension
            assert len(pooled_embedding.shape) == 2
            assert pooled_embedding.shape[0] == wav2vec_embedding.shape[0]

        # potentially apply masking
        # masking assumed shape [BS, EMBEDDING_DIM, TIME_DIM], so we
        # artificially add a time dimension of size 1 which is
        # immediately squeezed out after the masking operation
        masked_pooled_embedding = t.squeeze(
            self.embedding_masker(pooled_embedding[:, :, None])
        )

        if self.cfg.embedding_layer_idx < 0:
            return masked_pooled_embedding

        # loop over fc layers until we have reached the index
        # from which to select the speaker embedding
        x = masked_pooled_embedding

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

    def compute_ensemble_embedding(self, input_tensor: t.Tensor):
        # transform input
        # (of shape [BS, 1, NUM_AUDIO_SAMPLES] or [1, NUM_AUDIO_SAMPLES])
        # to the required [BS, NUM_AUDIO_SAMPLES]
        if len(input_tensor.shape) == 3 and input_tensor.shape[1] == 1:
            input_tensor = torch.squeeze(input_tensor)
        if len(input_tensor.shape) == 1:
            input_tensor = torch.stack([input_tensor])

        # first compute the wav2vec embeddings: will be shape [BS, 512, NUM_WINDOWS]
        wav2vec_embeddings = self.wav2vec.model(input_tensor, output_hidden_states=True)

        embeddings = []
        end = 13
        start = 13 - self.cfg.num_ensembles
        for idx, output in enumerate(wav2vec_embeddings.hidden_states[start:end]):
            pooled_output = self.stat_pooling(output)
            pooled_output = t.squeeze(pooled_output)
            embeddings.append(pooled_output)
            # print(idx)
            # print(f"{output.shape=}")
            # print(f"{pooled_output.shape=}")

        return embeddings
