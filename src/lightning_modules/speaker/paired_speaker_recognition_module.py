################################################################################
#
# Define a base lightning module for a speaker recognition network.
#
# Author(s): Nik Vaessen
################################################################################

import logging

from abc import abstractmethod
from typing import Callable, Optional, List, Any

import numpy as np
import torch as t
import torch.nn

import torchmetrics

from pytorch_lightning.core.decorators import auto_move_data
from omegaconf import DictConfig

from src.data.modules.speaker.training_batch_speaker import (
    PairedSpeakerClassificationDataBatch,
)
from src.eval_metrics import calculate_eer, calculate_mdc
from src.lightning_modules.base_lightning_module import BaseLightningModule

################################################################################
# Definition of speaker recognition API

# A logger for this file

log = logging.getLogger(__name__)


class PairedSpeakerRecognitionLightningModule(BaseLightningModule):
    def __init__(
        self,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        auto_lr_find: Optional[float] = None,
    ):
        super().__init__(hyperparameter_config, loss_fn_constructor, auto_lr_find)

        # used to keep track of training/val accuracy
        self.metric_train_acc = torchmetrics.Accuracy()
        self.metric_train_loss = torchmetrics.AverageMeter()
        self.metric_valid_acc = torchmetrics.Accuracy()

    @abstractmethod
    def compute_speaker_equality(
        self, wav_tensor: t.Tensor, other_wav_tensor
    ) -> t.Tensor:
        # transform two wav tensor with shape [BATCH_SIZE, N] and
        # [BATCH_SIZE, M], where M and N represent two different audio
        # sample lengths
        # into a speaker prediction of shape [BATCH_SIZE, 1] where
        # positive score means high likelihood of equality and
        # negative score means low likelihood of equality
        pass

    @auto_move_data
    def forward(self, input_tensor: torch.Tensor, other_input_tensor):
        scores = self.compute_speaker_equality(input_tensor, other_input_tensor)

        return scores

    def training_step(
        self,
        batch: PairedSpeakerClassificationDataBatch,
        batch_idx: int,
        optimized_idx: Optional[int] = None,
    ):
        primary_audio_input = batch.primary_network_input
        secondary_audio_input = batch.secondary_network_input
        label = batch.ground_truth

        equality_scores = self.compute_speaker_equality(
            primary_audio_input, secondary_audio_input
        )
        loss, prediction = self.loss_fn(equality_scores, label)

        with torch.no_grad():
            self._log_train_loss(loss, batch_idx)
            self._log_train_acc(prediction, label, batch_idx)

        return {"loss": loss}

    def _log_train_acc(self, prediction: t.Tensor, label: t.Tensor, batch_idx: int):
        self.metric_train_acc(prediction, label)

        if batch_idx % 100 == 0:
            self.log(
                "train_acc",
                self.metric_train_acc.compute(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.metric_train_acc.reset()

    def _log_train_loss(self, loss: t.Tensor, batch_idx: int):
        self.metric_train_loss(loss)

        if batch_idx % 100 == 0:
            self.log(
                "train_loss",
                self.metric_train_loss.compute(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.metric_train_loss.reset()

    def validation_step(
        self,
        batch: PairedSpeakerClassificationDataBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ):
        primary_audio_input = batch.primary_network_input
        secondary_audio_input = batch.secondary_network_input
        label = batch.ground_truth

        equality_scores = self.compute_speaker_equality(
            primary_audio_input, secondary_audio_input
        )
        loss, prediction = self.loss_fn(equality_scores, label)

        with torch.no_grad():
            self.metric_valid_acc(prediction, label)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "prediction": equality_scores.detach().cpu().numpy().tolist(),
            "label": label.detach().cpu().numpy().tolist(),
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        val_eer = self._evaluate(outputs)["eer"]
        self.log("val_eer", val_eer, prog_bar=True)

        self.log("val_acc", self.metric_valid_acc.compute(), prog_bar=True)
        self.metric_valid_acc.reset()

    def test_step(
        self,
        batch: PairedSpeakerClassificationDataBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ):
        if batch.batch_size != 1:
            raise ValueError("expecting a batch size of 1 for evaluation")

        primary_audio_input = batch.primary_network_input
        secondary_audio_input = batch.secondary_network_input
        label = batch.ground_truth

        equality_score = self.compute_speaker_equality(
            primary_audio_input, secondary_audio_input
        )

        return {
            "prediction": equality_score.detach().cpu().numpy().tolist(),
            "label": label.detach().cpu().numpy().tolist(),
        }

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self._evaluate(outputs))

    @staticmethod
    def _evaluate(outputs):
        # outputs is a list of dictionary with keys `label` and `prediction`
        # where `label` is int or list of ints with value 0 (when pair is not equaL)
        # or value 1 (when pair equal)
        # where `prediction` is int or list of ints with probability for being equal
        # (0 unlikely to be equal, 1 likely to be equal
        ground_truth_scores: List[int] = []
        prediction_scores: List[int] = []

        for d in outputs:
            label = d["label"]
            prediction = d["prediction"]

            if isinstance(label, int):
                ground_truth_scores.append(label)
            else:
                ground_truth_scores.extend(label)

            if isinstance(prediction, int):
                prediction_scores.append(prediction)
            else:
                prediction_scores.extend(prediction)

        # ground truth score should be 0 when the pair at a particular index
        # was `not equal`, and 1 when the pair was `equal`
        # prediction score is the probability that a pair at particular index
        # was equal, with 0 being very likely not equal and 1 being very likely equal
        # info statistics on ground-truth and prediction scores
        import pandas as pd

        # normalize scores to be between 0 and 1
        prediction_scores = np.clip((np.array(prediction_scores) + 1) / 2, 0, 1)
        prediction_scores = prediction_scores.tolist()

        print("ground truth scores")
        print(pd.DataFrame(ground_truth_scores).describe())
        print("prediction scores")
        print(pd.DataFrame(prediction_scores).describe())

        # compute EER
        try:
            eer, eer_threshold = calculate_eer(
                ground_truth_scores, prediction_scores, pos_label=1
            )

            if np.isnan(eer_threshold):
                eer = 1
        except (ValueError, ZeroDivisionError) as e:
            # if NaN values, we just return a very bad score
            # so that hparam searches don't crash
            print(f"eer calculation had {e}")

            eer = 1
            eer_threshold = 1337

        # compute mdc
        try:
            mdc, mdc_threshold = calculate_mdc(
                ground_truth_scores, prediction_scores
            )
            if isinstance(mdc_threshold, list):
                if len(mdc_threshold) == 0:
                    mdc_threshold = -1
                else:
                    mdc_threshold = mdc_threshold[0]
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
