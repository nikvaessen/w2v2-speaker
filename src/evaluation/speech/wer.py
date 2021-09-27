################################################################################
#
# Calculating word-error-rate
#
# Author(s): Nik Vaessen
################################################################################

from typing import List

from jiwer import wer

################################################################################
# wrapper around jiwer


def calculate_wer(transcriptions: List[str], ground_truths: List[str]):
    return wer(ground_truths, transcriptions)
