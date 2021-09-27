################################################################################
#
# Implement the wav2spk network from:
#
# Wav2Spk: A Simple DNN Architecture for Learning Speaker Embeddingsfrom Waveforms
#
# https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1287.pdf
#
# Author(s): Nik Vaessen
################################################################################

from dataclasses import dataclass
from typing import List, Optional, Callable

import speechbrain as sb
import torch as t
import torch.nn as nn

from omegaconf import DictConfig

from src.evaluation.speaker.speaker_recognition_evaluator import (
    EvaluationPair,
    SpeakerRecognitionEvaluator,
)
from src.layers.pooling import MeanStdStatPool1D, MeanStatPool1D
from src.layers.temporal_gating import TemporalGate
from src.lightning_modules.speaker.speaker_recognition_module import (
    SpeakerRecognitionLightningModule,
)
from src.optim.loss.aam_softmax import AngularAdditiveMarginSoftMaxLoss


################################################################################
# Implementation of wav2spk


@dataclass
class Wav2SpkModuleConfig:
    # settings for temporal gating
    apply_temporal_gating: bool

    # settings for fc head
    hidden_fc_layers_out: List[int]
    embedding_layer_idx: int
    stat_pooling_type: str


class Wav2SpkModule(SpeakerRecognitionLightningModule):
    def __init__(
        self,
        hyperparameters_to_save: DictConfig,
        cfg: Wav2SpkModuleConfig,
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

        # create stat_pool layer
        if self.cfg.stat_pooling_type == "mean":
            self.stat_pooling = MeanStatPool1D(dim_to_reduce=2)
        elif self.cfg.stat_pooling_type == "mean+std":
            self.stat_pooling = MeanStdStatPool1D(dim_to_reduce=2)
        else:
            raise ValueError(
                f"unknown value {cfg.stat_pooling_type=}, should be one of "
                f"['mean', 'mean+std']"
            )
        self.stat_pool_dimension = self._determine_stat_pool_embedding_size()

        # create feature encoder
        self.encoder = self._setup_feature_encoder()

        # create temporal gate
        self.temporal_gate = TemporalGate(512)

        # create feature aggregator
        self.aggregator = self._setup_feature_aggregator()

        # create fc layers
        self.fc_list = self._setup_fc_head()

        if isinstance(self.loss_fn, AngularAdditiveMarginSoftMaxLoss):
            raise ValueError("wav2spk does not support aam softmax")

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
            return 512  # 512 = output of aggregation layers
        elif self.cfg.stat_pooling_type == "mean+std":
            return 512 * 2  # 512 = output of aggregation layers
        else:
            raise ValueError(f"unknown value for {self.cfg.stat_pooling_type=}")

    def _setup_feature_encoder(self):
        def _layer(
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
        ):
            return nn.Sequential(
                t.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_size,),
                    stride=(stride,),
                    padding=(padding,),
                ),
                t.nn.InstanceNorm1d(
                    out_channels,
                ),
                t.nn.ReLU(),
            )

        conv0 = _layer(
            in_channels=1, out_channels=40, kernel_size=10, stride=5, padding=4
        )
        conv1 = _layer(
            in_channels=40, out_channels=200, kernel_size=5, stride=4, padding=2
        )
        conv2 = _layer(
            in_channels=200, out_channels=300, kernel_size=5, stride=2, padding=2
        )
        conv3 = _layer(
            in_channels=300, out_channels=512, kernel_size=3, stride=2, padding=1
        )
        conv4 = _layer(
            in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1
        )

        return nn.ModuleList([conv0, conv1, conv2, conv3, conv4])

    def _setup_feature_aggregator(self):
        def _layer(
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
        ):
            return nn.Sequential(
                t.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_size,),
                    stride=(stride,),
                    padding=(padding,),
                ),
                t.nn.ReLU(),
            )

        conv5 = _layer(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        conv6 = _layer(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        conv7 = _layer(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        conv8 = _layer(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )

        return nn.ModuleList([conv5, conv6, conv7, conv8])

    def _setup_fc_head(self):
        fc_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        in_features=self.stat_pool_dimension
                        if idx == 0
                        else self.cfg.hidden_fc_layers_out[idx - 1],
                        out_features=num_out,
                    ),
                    nn.ReLU(),
                )
                for idx, num_out in enumerate(self.cfg.hidden_fc_layers_out)
            ]
        )
        fc_list.append(
            nn.Sequential(
                nn.Linear(
                    in_features=self.stat_pool_dimension
                    if len(self.cfg.hidden_fc_layers_out) == 0
                    else self.cfg.hidden_fc_layers_out[-1],
                    out_features=self.num_speakers,
                ),
                sb.nnet.activations.Softmax(apply_log=True),
            )
        )

        return fc_list

    def generate_example_input(
        self, include_batch_dimension: bool, batch_size: Optional[int] = None
    ):
        if include_batch_dimension:
            # [BATCH_SIZE, NUMBER_OF_AUDIO_SAMPLES]
            # the `16000` varies depending on length of audio file
            # (1 second in this case)
            shape = [batch_size, 1, 16000]
        else:
            # [BATCH_SIZE, NUMBER_OF_AUDIO_SAMPLES]
            # the `16000` varies depending on length of audio file
            # (1 second in this case)
            shape = [1, 16000]

        return t.rand(size=shape)

    def _fc_head_ops_pre_spk_embedding(self, speech_embedding: t.tensor):
        # first apply stat pooling
        pooled_embedding = self.stat_pooling(speech_embedding)

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
        # first compute the wav2vec embeddings: out will have shape
        # [BS, 512, NUM_WINDOWS]
        embeddings = input_tensor

        for idx, conv in enumerate(self.encoder):
            embeddings = conv(embeddings)

        # we optionally apply temporal gating on the embeddings to act
        # as a learned VAD
        if self.cfg.apply_temporal_gating:
            gated_embeddings = self.temporal_gate(embeddings)
        else:
            gated_embeddings = embeddings

        # then we apply the feature aggregator: output will have shape
        # [BS, 512, NUM_WINDOWS]
        aggregated_embeddings = gated_embeddings

        for idx, conv in enumerate(self.aggregator):
            aggregated_embeddings = conv(aggregated_embeddings)

        # we end with all the operations to get to the speaker embeddings
        return self._fc_head_ops_pre_spk_embedding(aggregated_embeddings)

    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        # we apply all operations we need to apply on the speaker
        # embedding to get to the classification prediction
        prediction = self._fc_head_ops_post_spk_embedding(embedding_tensor)

        return prediction.squeeze()
