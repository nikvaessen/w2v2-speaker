################################################################################
#
# Implement the x-vector network for speaker recognition as a
# SpeakerRecognitionModule.
#
# Author(s): Nik Vaessen
################################################################################

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
from src.lightning_modules.speaker.speaker_recognition_module import (
    SpeakerRecognitionLightningModule,
)
from src.optim.loss.aam_softmax import AngularAdditiveMarginSoftMaxLoss

################################################################################
# Implementation of x-vector network


@dataclass
class XVectorModuleConfig:
    # optional explicit overwrite of embedding size (e.g if you
    # need to load finetuned weights but want to experiment with another
    # pooling type in the evaluation)
    explicit_stat_pool_embedding_size: Optional[int]
    explicit_num_speakers: Optional[int]

    tdnn_blocks: int = 5
    tdnn_channels: List[int] = field(default_factory=[512, 512, 512, 512, 1500])
    tdnn_kernel_sizes: List[int] = field(default_factory=[5, 3, 3, 1, 1])
    tdnn_dilations: List[int] = field(default_factory=[1, 2, 3, 1, 1])
    lin_neurons: int = 512
    in_channels: int = 40


class XVectorModule(SpeakerRecognitionLightningModule):
    def __init__(
        self,
        hyperparameters_to_save: DictConfig,
        cfg: XVectorModuleConfig,
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

        self.feature_extractor = Xvector(
            tdnn_blocks=cfg.tdnn_blocks,
            tdnn_channels=cfg.tdnn_channels,
            tdnn_kernel_sizes=cfg.tdnn_kernel_sizes,
            tdnn_dilations=cfg.tdnn_dilations,
            lin_neurons=cfg.lin_neurons,
            in_channels=cfg.in_channels,
        )
        self.classifier = Classifier(
            (None, None, self.embedding_size),
            out_neurons=self.num_speakers
            if cfg.explicit_num_speakers is None
            else cfg.explicit_num_speakers,
        )

        if isinstance(self.loss_fn, AngularAdditiveMarginSoftMaxLoss):
            raise ValueError("xvector does not support aam softmax")

    def generate_example_input(
        self, include_batch_dimension: bool, batch_size: Optional[int] = None
    ):
        if include_batch_dimension:
            # [BATCH_SIZE, NUMBER_OF_WINDOWS, NUMBER_OF_MODEL_COEFFICIENTS]
            # the `100` varies depending on length of audio file
            # the `40` can be replaced by any other number of mel coefficients
            shape = [batch_size, 100, 40]
        else:
            # [NUMBER_OF_WINDOWS, NUMBER_OF_MODEL_COEFFICIENTS]
            # the `100` varies depending on length of audio file
            # the `40` can be replaced by any other number of mel coefficients
            shape = [100, 40]

        return t.rand(size=shape)

    def compute_speaker_embedding(self, input_tensor: t.Tensor) -> t.Tensor:
        x = input_tensor

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
