################################################################################
#
# Wrap around implementation of binary cross-entropy loss.
#
# Author(s): Nik Vaessen
################################################################################

import torch as t
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# wrap around cross-entropy loss of PyTorch


class BinaryCrossEntropyLoss(t.nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits: t.Tensor, label_indexes: t.Tensor):
        return self._bce_loss(logits, label_indexes)

    def _bce_loss(self, logits: t.Tensor, label_indexes: t.Tensor):
        # logits (unnormalized quantities on which sigmoid is applied)
        # with shape [BATCH_SIZE, 1] and
        # label indexes (integers in {0, 1}) with shape [BATCH SIZE]
        logits = logits.squeeze().to(t.float32)
        label_indexes = label_indexes.squeeze().to(t.float32)

        loss = F.binary_cross_entropy_with_logits(logits, label_indexes)

        with t.no_grad():
            # put predictions into [0, 1] range for later calculation of accuracy
            prediction = t.sigmoid(logits).detach()

        return loss, prediction
