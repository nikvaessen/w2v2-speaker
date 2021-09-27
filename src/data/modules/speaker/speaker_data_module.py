################################################################################
#
# Abstract LightningDataModule for speaker recognition
#
# Author(s): Nik Vaessen
################################################################################

from abc import abstractmethod
from typing import List

import pytorch_lightning

from src.evaluation.speaker.speaker_recognition_evaluator import EvaluationPair


################################################################################
# abstract class of a lightning data module for speaker recognition


class SpeakerLightningDataModule(pytorch_lightning.LightningDataModule):
    @property
    @abstractmethod
    def num_speakers(self) -> int:
        pass

    @property
    @abstractmethod
    def val_pairs(self) -> List[EvaluationPair]:
        pass

    @property
    @abstractmethod
    def test_pairs(self) -> List[EvaluationPair]:
        pass

    @property
    @abstractmethod
    def summary(self):
        pass
