################################################################################
#
# Abstract LightningDataModule for speaker recognition
#
# Author(s): Nik Vaessen
################################################################################

from abc import abstractmethod
from typing import List

import pytorch_lightning

from src.tokenizer.base import BaseTokenizer

################################################################################
# abstract class of a lightning data module for speaker recognition


class SpeechLightningDataModule(pytorch_lightning.LightningDataModule):
    @property
    @abstractmethod
    def vocabulary(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def summary(self):
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> BaseTokenizer:
        pass
