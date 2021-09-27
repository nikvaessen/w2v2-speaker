################################################################################
#
# Implementation of Cross-entropy loss.
#
# Author(s): Nik Vaessen
################################################################################

import torch as t
import torch.nn.functional as F

################################################################################
# wrap around PyTorch cross-entropy loss implementation


class CrossEntropyLoss(t.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: t.Tensor, label_indexes: t.Tensor):
        return self._ce_loss(logits, label_indexes)

    def _ce_loss(self, logits: t.Tensor, label_indexes: t.Tensor):
        # logits (unnormalized quantities on which softmax is applied)
        # with shape [BATCH_SIZE, NUM_SPEAKERS] and
        # label indexes (integers in range [0, NUM_SPEAKERS-1])
        # with shape [BATCH SIZE]
        loss = F.cross_entropy(logits, label_indexes)

        with t.no_grad():
            # put predictions into [0, 1] range for later calculation of accuracy
            prediction = F.softmax(logits, dim=1).detach()

        return loss, prediction
