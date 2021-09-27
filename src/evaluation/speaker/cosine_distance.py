################################################################################
#
# Implement the cosine distance evaluation metric and evaluator.
#
# Author(s): Nik Vaessen
################################################################################

import random

from typing import List, Tuple

import torch
import torch as t
import tqdm

from torch.nn import CosineSimilarity

from src.evaluation.speaker.speaker_recognition_evaluator import (
    SpeakerRecognitionEvaluator,
    EmbeddingSample,
    compute_mean_std_batch,
    center_batch,
    length_norm_batch,
)


################################################################################
# Pytorch Module for cosine distance similarity score


class CosineDistanceSimilarityModule(t.nn.Module):
    def __init__(self):
        super().__init__()

        self.cosine_distance = CosineSimilarity()

    def forward(self, embedding: t.Tensor, other_embedding: t.Tensor):
        # we assume both inputs have dimensionality [batch_size, NUM_FEATURES]
        cos_dist = self.cosine_distance(embedding, other_embedding)

        # return a score between [-1, 1]
        return cos_dist


################################################################################
# wrap module in a method


def similarity_by_cosine_distance(
    embedding: t.Tensor, other_embedding: t.Tensor
) -> t.Tensor:
    """
    Compute a similarity score between 0 and 1 for two audio feature vectors.

    :param embedding: torch tensor of shape [BATCH_SIZE, N_FEATURES] representing a batch of speaker features
    :param other_embedding: torch tensor of shape [BATCH_SIZE< N_FEATURES] representing another batch of speaker features
    :return: a tensor containing scores close to 1 if embeddings contain same speaker, close to 0 otherwise
    """
    return CosineDistanceSimilarityModule()(embedding, other_embedding)


################################################################################
# Implement an evaluator based on cosine distance


class CosineDistanceEvaluator(SpeakerRecognitionEvaluator):
    def __init__(
        self,
        center_before_scoring: bool,
        length_norm_before_scoring: bool,
        max_num_training_samples: int,
    ):
        super().__init__(max_num_training_samples)

        self.center_before_scoring = center_before_scoring
        self.length_norm_before_scoring = length_norm_before_scoring

        # set in self#fit_parameters
        self.mean = None
        self.std = None

    def fit_parameters(
        self, embedding_tensors: List[t.Tensor], _label_tensors: List[t.Tensor]
    ):
        # note we don't need label tensors
        if not self._using_parameters():
            return

        if len(embedding_tensors) <= 2:
            raise ValueError("mean/std calculation requires more than 2 samples")

        # create a tensor of shape [BATCH_SIZE*len(tensors), EMBEDDING_SIZE]
        all_tensors = t.stack(embedding_tensors, dim=0)

        self.mean, self.std = compute_mean_std_batch(all_tensors)

    def reset_parameters(self):
        if not self._using_parameters():
            return

        self.mean = None
        self.std = None

    def _using_parameters(self):
        return self.center_before_scoring

    def _compute_prediction_scores(
        self, pairs: List[Tuple[EmbeddingSample, EmbeddingSample]]
    ) -> List[float]:
        # dirty hack to recognize ensemble of embeddings and change behaviour
        if isinstance(pairs[0][0].embedding, list):
            return self._compute_ensemble_prediction_scores(pairs)

        # dirty hack to recognize `non_pooled embedding` and change behaviour
        if len(pairs[0][0].embedding.shape) == 2:
            return self._compute_non_pooled_prediction_scored(pairs)

        left_samples, right_samples = self._transform_pairs_to_tensor(pairs)

        if self.center_before_scoring:
            print(f"centering with {self.mean=} and {self.std=}")
            left_samples = center_batch(left_samples, self.mean, self.std)
            right_samples = center_batch(right_samples, self.mean, self.std)

        if self.length_norm_before_scoring:
            print("applying length norm...")
            left_samples = length_norm_batch(left_samples)
            right_samples = length_norm_batch(right_samples)

        scores = compute_cosine_scores(left_samples, right_samples)

        return scores

    def _compute_ensemble_prediction_scores(
        self, pairs: List[Tuple[EmbeddingSample, EmbeddingSample]]
    ) -> List[float]:
        print("detected an ensemble of embedding tensors!")
        # extract each ensemble pair
        num_ensembles = len(pairs[0][0].embedding)
        ensembles = []

        for i in range(num_ensembles):
            ensemble_pairs = []

            for p in pairs:
                s1, s2 = p

                if not isinstance(s1.embedding, list) or not isinstance(
                    s2.embedding, list
                ):
                    raise ValueError("not every embedding sample is an ensemble")

                if (
                    not len(s1.embedding) == num_ensembles
                    or not len(s2.embedding) == num_ensembles
                ):
                    raise ValueError(f"expected each list to have len {num_ensembles=}")

                s1 = EmbeddingSample(sample_id=s1.sample_id, embedding=s1.embedding[i])
                s2 = EmbeddingSample(sample_id=s2.sample_id, embedding=s2.embedding[i])

                ensemble_pairs.append((s1, s2))

            ensembles.append(ensemble_pairs)

        scores = [self._compute_prediction_scores(ens) for ens in ensembles]

        assert len(scores) == num_ensembles
        for score_list in scores:
            assert len(score_list) == len(pairs)

        # linearly combine all scores
        combined_scores = []

        for idx, p in enumerate(pairs):
            score = 0

            for ensemble_idx in range(num_ensembles):
                score += scores[ensemble_idx][idx] * (1 / num_ensembles)

            combined_scores.append(score)

        assert len(combined_scores) == len(pairs)

        return combined_scores

    def _compute_non_pooled_prediction_scored(
        self, pairs: List[Tuple[EmbeddingSample, EmbeddingSample]]
    ) -> List[float]:
        # we don't care about centering/length norm anyway
        scores = []

        with tqdm.tqdm(total=len(pairs)) as p:
            for left, right in pairs:
                scores.append(
                    compute_non_pooled_cosine_scores(left.embedding, right.embedding)
                )

                p.update(1)

        return scores


def compute_non_pooled_cosine_scores(left_tensor, right_tensor) -> float:
    # assume left tensor to have shape [p1, embedding_size] and
    #       right tensor to have shape [p2, embedding_size]
    p1 = left_tensor.shape[0]
    p2 = right_tensor.shape[0]

    # select a subset of 50 tensors to compare
    left_tensor = left_tensor[random.sample(range(p1), min(50, p1)), :]
    right_tensor = right_tensor[random.sample(range(p2), min(50, p2)), :]

    # left tensor needs to be interleaved p2 times
    left_interleaved = torch.repeat_interleave(
        left_tensor, right_tensor.shape[0], dim=0
    )

    # right needs to be repeated p1 times
    right_repeat = right_tensor.repeat(left_tensor.shape[0], 1)

    assert left_interleaved.shape == right_repeat.shape

    with t.no_grad():
        try:
            left_interleaved_gpu = left_interleaved.cuda()
            right_repeat_gpu = right_repeat.cuda()
            score_tensor = similarity_by_cosine_distance(
                left_interleaved_gpu, right_repeat_gpu
            )
        except:
            score_tensor = similarity_by_cosine_distance(left_interleaved, right_repeat)

    return t.mean(score_tensor).detach().cpu().numpy().tolist()


def compute_cosine_scores(
    left_samples: t.Tensor, right_samples: t.Tensor
) -> List[float]:
    # compute the scores
    score_tensor = similarity_by_cosine_distance(left_samples, right_samples)

    return score_tensor.detach().cpu().numpy().tolist()
