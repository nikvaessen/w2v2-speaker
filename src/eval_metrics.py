################################################################################
#
# This file implements two quantitative measures for speaker identification:
#
# * equal error rate
# * minimum detection cost
#
# It also provides a CLI for calculating these measures on some
# predefined pairs of speaker (mis)matches.
#
# Author(s): Nik Vaessen
################################################################################

from operator import itemgetter
from typing import List, Tuple

import numpy as np

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

################################################################################
# helper methods for both measures


def _verify_correct_scores(
    groundtruth_scores: List[int], predicted_scores: List[float]
):
    if len(groundtruth_scores) != len(predicted_scores):
        raise ValueError(
            f"length of input lists should match, while"
            f" groundtruth_scores={len(groundtruth_scores)} and"
            f" predicted_scores={len(predicted_scores)}"
        )
    # if np.min(predicted_scores) < 0 or np.max(predicted_scores) > 1:
    #     raise ValueError(
    #         f"predictions should be in range [0, 1], while they"
    #         f" are actually in range "
    #         f"[{np.min(predicted_scores)}, "
    #         f"{np.max(predicted_scores)}]"
    #     )
    if not all(np.isin(groundtruth_scores, [0, 1])):
        raise ValueError(
            f"groundtruth values should be either 0 and 1, while "
            f"they are actually one of {np.unique(groundtruth_scores)}"
        )


################################################################################
# EER (equal-error-rate)


def calculate_eer(
    groundtruth_scores: List[int], predicted_scores: List[float], pos_label: int = 1
):
    """
    Calculate the equal error rate between a list of groundtruth pos/neg scores
    and a list of predicted pos/neg scores.

    Adapted from: https://github.com/a-nagrani/VoxSRC2020/blob/master/compute_EER.py

    :param groundtruth_scores: a list of groundtruth integer values (either 0 or 1)
    :param predicted_scores: a list of prediction float values (in range [0, 1])
    :param pos_label: which value (either 0 or 1) represents positive. Defaults to 1
    :return: a tuple containing the equal error rate and the corresponding threshold
    """
    _verify_correct_scores(groundtruth_scores, predicted_scores)

    if not all(np.isin([pos_label], [0, 1])):
        raise ValueError(f"The positive label should be either 0 or 1, not {pos_label}")

    fpr, tpr, thresholds = roc_curve(
        groundtruth_scores, predicted_scores, pos_label=pos_label
    )
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer).item()

    return eer, thresh


################################################################################
# minimum detection cost - taken from
# https://github.com/a-nagrani/VoxSRC2020/blob/master/compute_min_dcf.py
# Copyright 2018  David Snyder
# This script is modified from the Kaldi toolkit -
# https://github.com/kaldi-asr/kaldi/blob/8ce3a95761e0eb97d95d3db2fcb6b2bfb7ffec5b/egs/sre08/v1/sid/compute_min_dcf.py


def _compute_error_rates(
    groundtruth_scores: List[int],
    predicted_scores: List[float],
) -> Tuple[List[float], List[float], List[float]]:
    """
    Creates a list of false-negative rates, a list of false-positive rates
    and a list of decision thresholds that give those error-rates.

    :param groundtruth_scores: a list of groundtruth integer values (either 0 or 1)
    :param predicted_scores: a list of prediction float values (in range [0, 1])
    :return: a triple with a list of false negative rates, false positive rates
     and a list of decision threshold
    for those rates.
    """
    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(
        *sorted(
            [(index, threshold) for index, threshold in enumerate(predicted_scores)],
            key=itemgetter(1),
        )
    )

    groundtruth_scores = [groundtruth_scores[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(groundtruth_scores)):
        if i == 0:
            fnrs.append(groundtruth_scores[i])
            fprs.append(1 - groundtruth_scores[i])
        else:
            fnrs.append(fnrs[i - 1] + groundtruth_scores[i])
            fprs.append(fprs[i - 1] + 1 - groundtruth_scores[i])
    fnrs_norm = sum(groundtruth_scores)
    fprs_norm = len(groundtruth_scores) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of correct positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]

    return fnrs, fprs, thresholds


def _compute_min_dfc(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    """
    Computes the minimum of the detection cost function. The comments refer to
    equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.

    :param fnrs: the list of false negative rates
    :param fprs: the list of false positive rates
    :param thresholds: the list of decision thresholds
    :param p_target: a priori probability of the specified target speaker
    :param c_miss: cost of a missed detection
    :param c_fa: cost of a spurious detection
    :return: the minimum detection cost and accompanying threshold
    """
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]

    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]

    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    return min_dcf, min_c_det_threshold


def calculate_mdc(
    groundtruth_scores: List[int],
    predicted_scores: List[float],
    c_miss: float = 1,
    c_fa: float = 1,
    p_target: float = 0.05,
):
    """
    Calculate the minimum detection cost and threshold based on a list of
    groundtruth and prediction pairs.

    :param groundtruth_scores: the list of groundtruth scores
    :param predicted_scores:
    :param groundtruth_scores: a list of groundtruth integer values (either 0 or 1)
    :param predicted_scores: a list of prediction float values (in range [0, 1])
    :param p_target: a priori probability of the specified target speaker
    :param c_miss: cost of a missed detection
    :param c_fa: cost of a spurious detection
    :return: a tuple containing the minimum detection score and the corresponding threshold
    """
    _verify_correct_scores(groundtruth_scores, predicted_scores)
    if c_miss < 1:
        raise ValueError(f"c_miss={c_miss} should be >= 1")
    if c_fa < 1:
        raise ValueError(f"c_fa={c_fa} should be >= 1")
    if p_target < 0 or p_target > 1:
        raise ValueError(f"p_target={p_target} should be between 0 and 1")

    fnrs, fprs, thresholds = _compute_error_rates(groundtruth_scores, predicted_scores)
    mindcf, threshold = _compute_min_dfc(fnrs, fprs, thresholds, p_target, c_miss, c_fa)

    return mindcf, threshold
