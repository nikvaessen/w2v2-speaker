################################################################################
#
# Implements a dummy module which uses very few parameters to generate
# predictions/embeddings.
# This is useful for debugging training schedules as it removes the
# heavy computation for each step so a full training run can be executed
# fairly quickly.
#
# Author(s): Nik Vaessen
################################################################################

from dataclasses import dataclass
from typing import List, Optional, Callable

import torch as t

from omegaconf import DictConfig

from src.evaluation.speaker.speaker_recognition_evaluator import (
    EvaluationPair,
    SpeakerRecognitionEvaluator,
)
from src.lightning_modules.speaker.speaker_recognition_module import (
    SpeakerRecognitionLightningModule,
)

################################################################################
# Implementation of a very light-weight neural network


@dataclass
class DummyModuleConfig:
    pass


class DummyModule(SpeakerRecognitionLightningModule):
    def __init__(
        self,
        hyperparameters_to_save: DictConfig,
        cfg: DummyModuleConfig,
        num_speakers: int,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        validation_pairs: List[EvaluationPair],
        test_pairs: List[EvaluationPair],
        evaluator: SpeakerRecognitionEvaluator,
    ):
        super().__init__(
            hyperparameter_config=hyperparameters_to_save,
            num_speakers=num_speakers,
            embedding_size=2,
            loss_fn_constructor=loss_fn_constructor,
            validation_pairs=validation_pairs,
            test_pairs=test_pairs,
            evaluator=evaluator,
            embeddings_are_pooled=True
        )

        self.cfg = cfg

        # just create a parameter so optimizer doesn't complain
        self.fc1 = t.nn.Linear(in_features=2, out_features=num_speakers)

    def generate_example_input(
        self, include_batch_dimension: bool, batch_size: Optional[int] = None
    ):
        # any input works really
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
        std, mean = t.std_mean(input_tensor, dim=(1, 2))
        embedding = t.stack([mean, std]).t()

        return embedding

    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        prediction = self.fc1(embedding_tensor)

        return prediction
