################################################################################
#
# Base API for preprocessors
#
# Author(s): Nik Vaessen
################################################################################

from abc import abstractmethod
from typing import List, Union

from src.data.common import DebugWriter
from src.data.modules.speaker.training_batch_speaker import (
    SpeakerClassificationDataSample,
)


################################################################################
# base preprocessor


class Preprocessor:
    @abstractmethod
    def process(
        self, sample: SpeakerClassificationDataSample
    ) -> Union[SpeakerClassificationDataSample, List[SpeakerClassificationDataSample]]:
        # process a sample in a particular way and generate one or more
        # new samples
        pass

    @abstractmethod
    def init_debug_writer(
        self,
    ) -> DebugWriter:
        pass
