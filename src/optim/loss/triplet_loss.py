################################################################################
#
# Implement a wrapper around triplet loss for speaker recognition embeddings
#
# Author(s): Nik Vaessen
################################################################################

import random

from typing import List

import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# wrapper of triplet loss


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1):
        super().__init__()

        self.margin = margin

    def forward(self, embeddings: t.Tensor, label_indexes: t.Tensor):
        return self._triplet_loss(embeddings, label_indexes)

    def _triplet_loss(self, embeddings: t.Tensor, label_indexes: t.Tensor):
        # embeddings with shape [BATCH_SIZE, EMBEDDING_SIZE] and
        # label indexes (integers in range [0, NUM_SPEAKERS-1])
        # with shape [BATCH SIZE]

        # make sure we can generate triplets for each label
        with torch.no_grad():
            label_list: List[int] = label_indexes.detach().cpu().numpy().tolist()

        self.verify_labels(label_list)

        # generate a triplet for each batch dimension
        anchors = []
        positives = []
        negatives = []

        for batch_dim in range(embeddings.shape[0]):
            # get anchor
            label = label_indexes[batch_dim]
            anchor = embeddings[batch_dim].squeeze()

            # find positive
            positive = self._find_positive(
                embeddings=embeddings,
                label_list=label_list,
                label=label,
                exclude_idx=batch_dim,
            )

            # find negative
            negative = self._find_negative(
                embeddings=embeddings, label_list=label_list, label=label
            )

            # save anchor, positive, negative tuple
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        return F.triplet_margin_loss(
            anchor=t.stack(anchors),
            positive=t.stack(positives),
            negative=t.stack(negatives),
            margin=self.margin,
        )

    @staticmethod
    def cosine_distance(a: t.Tensor, b: t.Tensor):
        return 1 - t.div(F.cosine_similarity(a, b) + 1, 2)

    @staticmethod
    def _find_positive(
        embeddings: t.Tensor, label_list: List[int], label: int, exclude_idx: int
    ) -> t.Tensor:
        candidate_indexes = [
            idx for idx, l in enumerate(label_list) if label == l and exclude_idx != idx
        ]

        idx = random.choice(candidate_indexes)

        return embeddings[idx].squeeze()

    @staticmethod
    def _find_negative(
        embeddings: t.Tensor, label_list: List[int], label: int
    ) -> t.Tensor:
        candidate_indexes = [idx for idx, l in enumerate(label_list) if label != l]

        idx = random.choice(candidate_indexes)

        return embeddings[idx].squeeze()

    @staticmethod
    def verify_labels(label_list: List[int]):
        unique_labels = set(label_list)

        for label in unique_labels:
            assert label_list.count(label) >= 2
