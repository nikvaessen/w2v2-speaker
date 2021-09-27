################################################################################
#
# Implement the ecapa-tdnn network for speaker recognition as a
# SpeakerRecognitionModule.
#
# Author(s): Nik Vaessen
################################################################################

from dataclasses import dataclass
from typing import List, Optional, Callable

import torch
import torch as t

from omegaconf import DictConfig
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, Classifier

from src.evaluation.speaker.speaker_recognition_evaluator import (
    SpeakerRecognitionEvaluator,
    EvaluationPair,
)
from src.lightning_modules.speaker.speaker_recognition_module import (
    SpeakerRecognitionLightningModule,
)
from src.optim.loss.aam_softmax import AngularAdditiveMarginSoftMaxLoss

################################################################################
# Implementation of x-vector network


@dataclass
class EcapaTDNNModuleConfig:
    input_mel_coefficients: int
    lin_neurons: int
    channels: List[int]
    kernel_sizes: List[int]
    dilations: List[int]
    attention_channels: int
    res2net_scale: int
    se_channels: int
    global_context: bool
    pretrained_weights_path: Optional[str]

    # optional explicit overwrite of embedding size (e.g if you
    # need to load finetuned weights but want to experiment with another
    # pooling type in the evaluation)
    explicit_stat_pool_embedding_size: Optional[int]
    explicit_num_speakers: Optional[int]


class EcapaTdnnModule(SpeakerRecognitionLightningModule):
    def __init__(
        self,
        hyperparameters_to_save: DictConfig,
        cfg: EcapaTDNNModuleConfig,
        num_speakers: int,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        validation_pairs: List[EvaluationPair],
        test_pairs: List[EvaluationPair],
        evaluator: SpeakerRecognitionEvaluator,
    ):
        super().__init__(
            hyperparameter_config=hyperparameters_to_save,
            num_speakers=num_speakers,
            embedding_size=cfg.lin_neurons,
            loss_fn_constructor=loss_fn_constructor,
            validation_pairs=validation_pairs,
            test_pairs=test_pairs,
            evaluator=evaluator,
            embeddings_are_pooled=True,
        )

        self.cfg = cfg

        self.feature_extractor = ECAPA_TDNN(
            input_size=self.cfg.input_mel_coefficients,
            lin_neurons=self.cfg.lin_neurons,
            channels=self.cfg.channels,
            kernel_sizes=self.cfg.kernel_sizes,
            dilations=self.cfg.dilations,
            attention_channels=self.cfg.attention_channels,
            res2net_scale=self.cfg.res2net_scale,
            se_channels=self.cfg.se_channels,
            global_context=self.cfg.global_context,
        )
        self.classifier = Classifier(
            self.embedding_size,
            out_neurons=self.num_speakers
            if self.cfg.explicit_num_speakers is None
            else self.cfg.explicit_num_speakers,
        )

        if self.cfg.pretrained_weights_path is not None:
            self.feature_extractor.load_state_dict(
                t.load(self.cfg.pretrained_weights_path)
            )

        self.skip_classifier = isinstance(
            self.loss_fn, AngularAdditiveMarginSoftMaxLoss
        )

    def generate_example_input(
        self, include_batch_dimension: bool, batch_size: Optional[int] = None
    ):
        if include_batch_dimension:
            # [BATCH_SIZE, NUMBER_OF_WINDOWS, NUMBER_OF_MODEL_COEFFICIENTS]
            # the `100` varies depending on length of audio file
            # the `40` can be replaced by any other number of mel coefficients
            shape = [batch_size, 100, self.cfg.input_mel_coefficients]
        else:
            # [NUMBER_OF_WINDOWS, NUMBER_OF_MODEL_COEFFICIENTS]
            # the `100` varies depending on length of audio file
            # the `40` can be replaced by any other number of mel coefficients
            shape = [100, self.cfg.input_mel_coefficients]

        return t.rand(size=shape)

    def compute_speaker_embedding(self, input_tensor: t.Tensor) -> t.Tensor:
        embeddings = self.feature_extractor(input_tensor)

        if len(embeddings.shape) == 3:
            embeddings = embeddings.squeeze()

        if len(embeddings.shape) == 1:
            embeddings = torch.stack([embeddings])

        return embeddings

    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        if self.skip_classifier:
            return embedding_tensor.squeeze()

        # classifier expect shape [BATCH_SIZE, 1, EMBEDDING_SIZE]
        embedding_tensor = embedding_tensor[:, None, :]
        prediction = self.classifier(embedding_tensor)

        return prediction.squeeze()
