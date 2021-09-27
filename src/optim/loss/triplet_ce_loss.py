################################################################################
#
# Implement a wrapper around triplet loss and cross-entropy loss
# for speaker recognition embeddings
#
# Author(s): Nik Vaessen
################################################################################

import torch as t

from src.optim.loss.cross_entropy import CrossEntropyLoss
from src.optim.loss.triplet_loss import TripletLoss

################################################################################
# wrapper combining cross-entropy and triplet loss


class TripletCrossEntropyLoss(TripletLoss, CrossEntropyLoss):
    def __init__(self, c_ce: float = 1, c_triplet: float = 1):
        super().__init__()

        if c_ce < 1 or c_triplet < 1:
            raise ValueError(
                f"constants need to be natural numbers, while" f"{c_ce=}, {c_triplet=}"
            )

        self.c_ce = c_ce
        self.c_triplet = c_triplet

    def forward(self, embeddings: t.Tensor, logits: t.Tensor, label_indexes: t.Tensor):
        ce_loss, prediction = self._ce_loss(logits, label_indexes)
        triplet_loss = self._triplet_loss(embeddings, label_indexes)

        loss = self.c_ce * ce_loss + self.c_triplet * triplet_loss

        return loss, prediction
