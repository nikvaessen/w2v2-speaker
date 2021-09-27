################################################################################
#
# CTC loss for speech recognition
#
# Author(s): Nik Vaessen
################################################################################

import torch as t

import torch.nn as nn
import torch.nn.functional as F

################################################################################
# wrapper around ctc loss of pytorch


class CtcLoss(nn.Module):
    def __init__(self, blank_idx: int = 0):
        super().__init__()

        self.blank_idx = blank_idx

    def forward(
        self,
        predictions: t.Tensor,
        prediction_lengths: t.Tensor,
        ground_truths: t.Tensor,
        ground_truth_lengths: t.Tensor,
    ):
        original_device = predictions.device
        assert original_device == predictions.device == ground_truths.device

        # predictions will be shape [BATCH_SIZE, MAX_INPUT_SEQUENCE_LENGTH, CLASSES]
        # expected to be [MAX_INPUT_SEQUENCE_LENGTH, BATCH_SIZE, CLASSES] for
        # loss function
        predictions = t.transpose(predictions, 0, 1)

        # they also need to be log probabilities
        predictions = F.log_softmax(predictions, dim=2)

        # prediction lengths will be shape [BATCH_SIZE]
        pass  # already OK

        # ground truths will be shape [BATCH_SIZE, MAX_TARGET_SEQUENCE_LENGTH]
        pass  # already OK

        # ground_truth_lengths will be shape [BATCH_SIZE]
        pass  # already OK

        # ctc loss expects every tensor to be on CPU
        return F.ctc_loss(
            log_probs=predictions.to("cpu"),
            targets=ground_truths.to("cpu"),
            input_lengths=prediction_lengths.to("cpu"),
            target_lengths=ground_truth_lengths.to("cpu"),
            blank=self.blank_idx,
            zero_infinity=True,  # prevents any weird crashes
        ).to(original_device)
