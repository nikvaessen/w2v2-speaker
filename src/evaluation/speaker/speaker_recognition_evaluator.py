################################################################################
#
# Implement an Evaluator object which encapsulates the process
# computing performance metric of speech recognition task.
#
# Author(s): Nik Vaessen
################################################################################

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union
from warnings import warn

import numpy as np
import torch as t
import pandas as pd

from src.eval_metrics import calculate_eer, calculate_mdc
from torch.nn.functional import normalize

################################################################################
# define data structures required for evaluating


@dataclass
class EvaluationPair:
    same_speaker: bool
    sample1_id: str
    sample2_id: str


@dataclass
class EmbeddingSample:
    sample_id: str
    embedding: Union[t.Tensor, List[t.Tensor]]


################################################################################
# abstract base class of an evaluator


class SpeakerRecognitionEvaluator:
    def __init__(self, max_num_training_samples: int):
        self.max_num_training_samples = max_num_training_samples

    def evaluate(self, pairs: List[EvaluationPair], samples: List[EmbeddingSample]):
        # create a hashmap for quicker access to samples based on key
        sample_map = {}

        for sample in samples:
            if sample.sample_id in sample_map:
                raise ValueError(f"duplicate key {sample.sample_id}")

            sample_map[sample.sample_id] = sample

        # compute a list of ground truth scores and prediction scores
        ground_truth_scores = []
        prediction_pairs = []

        for pair in pairs:
            if pair.sample1_id not in sample_map or pair.sample2_id not in sample_map:
                warn(f"{pair.sample1_id} or {pair.sample2_id} not in sample_map")
                return {
                    "eer": -1,
                    "eer_threshold": -1,
                    "mdc": -1,
                    "mdc_threshold": -1,
                }

            s1 = sample_map[pair.sample1_id]
            s2 = sample_map[pair.sample2_id]

            gt = 1 if pair.same_speaker else 0

            ground_truth_scores.append(gt)
            prediction_pairs.append((s1, s2))

        prediction_scores = self._compute_prediction_scores(prediction_pairs)

        # normalize scores to be between 0 and 1
        prediction_scores = np.clip((np.array(prediction_scores) + 1) / 2, 0, 1)
        prediction_scores = prediction_scores.tolist()

        # info statistics on ground-truth and prediction scores
        print("ground truth scores")
        print(pd.DataFrame(ground_truth_scores).describe())
        print("prediction scores")
        print(pd.DataFrame(prediction_scores).describe())

        # compute EER
        try:
            eer, eer_threshold = calculate_eer(
                ground_truth_scores, prediction_scores, pos_label=1
            )
        except (ValueError, ZeroDivisionError) as e:
            # if NaN values, we just return a very bad score
            # so that hparam searches don't crash
            print(f"EER calculation had {e}")
            eer = 1
            eer_threshold = 1337

        # compute mdc
        try:
            mdc, mdc_threshold = calculate_mdc(ground_truth_scores, prediction_scores)
        except (ValueError, ZeroDivisionError) as e:
            print(f"mdc calculation had {e}")
            mdc = 1
            mdc_threshold = 1337

        return {
            "eer": eer,
            "eer_threshold": eer_threshold,
            "mdc": mdc,
            "mdc_threshold": mdc_threshold,
        }

    @abstractmethod
    def _compute_prediction_scores(
        self, pairs: List[Tuple[EmbeddingSample, EmbeddingSample]]
    ) -> List[float]:
        pass

    def _transform_pairs_to_tensor(
        self, pairs: List[Tuple[EmbeddingSample, EmbeddingSample]]
    ):
        # construct the comparison batches
        b1 = []
        b2 = []

        for s1, s2 in pairs:
            b1.append(s1.embedding)
            b2.append(s2.embedding)

        b1 = t.stack(b1)
        b2 = t.stack(b2)

        return b1, b2

    @abstractmethod
    def fit_parameters(
        self, embedding_tensors: List[t.Tensor], label_tensors: List[t.Tensor]
    ):
        pass

    @abstractmethod
    def reset_parameters(self):
        pass


################################################################################
# Utility methods common between evaluators


def compute_mean_std_batch(all_tensors: t.Tensor):
    # compute mean and std over each dimension of EMBEDDING_SIZE
    # with a tensor of shape [NUM_SAMPLES, EMBEDDING_SIZE]
    std, mean = t.std_mean(all_tensors, dim=0)

    return mean, std


def center_batch(embedding_tensor: t.Tensor, mean: t.Tensor, std: t.Tensor):
    # center the batch with shape [NUM_PAIRS, EMBEDDING_SIZE]
    # using the computed mean and std
    centered = (embedding_tensor - mean) / (std + 1e-12)

    return centered


def length_norm_batch(embedding_tensor: t.Tensor):
    # length normalize the batch with shape [NUM_PAIRS, EMBEDDING_SIZE]
    return normalize(embedding_tensor, dim=1)
