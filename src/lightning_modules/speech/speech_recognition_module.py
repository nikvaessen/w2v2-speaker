################################################################################
#
# Define a base lightning module for a speech recognition network.
#
# Author(s): Nik Vaessen
################################################################################

import logging

from abc import abstractmethod
from functools import reduce
from operator import mul
from typing import Callable, Optional, List, Dict, Tuple

import torch as t
import torch.nn
import torchmetrics

from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig

from src.data.modules.speech.training_batch_speech import SpeechRecognitionDataBatch
from src.evaluation.speech.wer import calculate_wer
from src.lightning_modules.base_lightning_module import BaseLightningModule
from src.optim.loss.ctc_loss import CtcLoss
from src.tokenizer.base import BaseTokenizer

################################################################################
# Definition of speaker recognition API

# A logger for this file

log = logging.getLogger(__name__)


class SpeechRecognitionLightningModule(BaseLightningModule):
    def __init__(
        self,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        tokenizer: BaseTokenizer,
        auto_lr_find: Optional[float] = None,
    ):
        super().__init__(hyperparameter_config, loss_fn_constructor, auto_lr_find)

        if not isinstance(self.loss_fn, CtcLoss):
            raise ValueError(
                f"expected loss class {CtcLoss}, " f"got {self.loss_fn.__class__}"
            )

        # required for decoding to text
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocabulary_size()

        # keep track of metrics
        self.metric_train_loss = torchmetrics.AverageMeter()
        self.metric_train_wer = torchmetrics.AverageMeter()

        self.metric_val_loss_clean = torchmetrics.AverageMeter()
        self.metric_val_loss_other = torchmetrics.AverageMeter()

        # set on first call to self#_log_transcription_progress step
        self.tracking_audio_sample: t.Tensor = None
        self.tracking_transcription: str = None
        self.tracking_sequence_length: int = None

    @abstractmethod
    def compute_speech_embedding(
        self, input_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        # transform:
        # 1) input_tensor with shape [BATCH_SIZE, NUM_SAMPLES]
        # 2) where 0:lengths[BATCH_IDX] are non-padded frames
        # into:
        # 1) an embedding of shape [BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE]
        # 2) a list of lengths which represents frames which are (non-padded)
        #    lengths (index 0:length_value is non-padded)
        pass

    @abstractmethod
    def compute_vocabulary_prediction(
        self, embedding_tensor: t.Tensor, lengths: List[int]
    ) -> Tuple[t.Tensor, List[int]]:
        # transform embedding tensor with shape [BATCH_SIZE, EMBEDDING_SIZE]
        # and list of non-padded range for each batch dimension
        # into a speaker prediction of shape [BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE]
        # and a list of non-padded range for each batch dimension
        pass

    @auto_move_data
    def forward(self, input_tensor: torch.Tensor, lengths: List[int]):
        embedding, emb_lengths = self.compute_speech_embedding(input_tensor, lengths)
        prediction, pred_lengths = self.compute_vocabulary_prediction(
            embedding, emb_lengths
        )

        return (embedding, emb_lengths), (prediction, pred_lengths)

    def training_step(
        self,
        batch: SpeechRecognitionDataBatch,
        batch_idx: int,
        optimized_idx: Optional[int] = None,
    ) -> STEP_OUTPUT:
        _, (
            letter_prediction,
            letter_prediction_lengths,
        ) = self.forward(batch.network_input, batch.input_lengths)

        loss = self.loss_fn(
            predictions=letter_prediction,
            ground_truths=batch.ground_truth,
            prediction_lengths=t.IntTensor(letter_prediction_lengths),
            ground_truth_lengths=batch.ground_truth_sequence_length,
        )

        with torch.no_grad():
            transcription = self._decode_letter_prediction(
                letter_prediction, letter_prediction_lengths
            )
            ground_truth_transcription = batch.ground_truth_strings
            train_wer = calculate_wer(transcription, ground_truth_transcription)

            # log training loss
            self.metric_train_loss(loss.detach().cpu().item())
            self.metric_train_wer(train_wer)

            if batch_idx % 100 == 0:
                self.log_dict(
                    {
                        "train_loss": self.metric_train_loss.compute(),
                        "train_wer": self.metric_train_wer.compute(),
                    },
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )

                self.metric_train_loss.reset()
                self.metric_train_wer.reset()

            if self.global_step % 1000 == 0:
                self._log_transcription_progress(batch)

        return loss

    def validation_step(
        self,
        batch: SpeechRecognitionDataBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        _, (
            letter_prediction,
            letter_prediction_lengths,
        ) = self.forward(batch.network_input, batch.input_lengths)

        loss = self.loss_fn(
            predictions=letter_prediction,
            ground_truths=batch.ground_truth,
            prediction_lengths=t.IntTensor(letter_prediction_lengths),
            ground_truth_lengths=batch.ground_truth_sequence_length,
        )

        with torch.no_grad():
            transcription = self._decode_letter_prediction(
                letter_prediction, letter_prediction_lengths
            )
            ground_truth_transcription = batch.ground_truth_strings

        return {
            "val_loss": loss,
            "transcription": transcription,
            "ground_truth": ground_truth_transcription,
        }

    def validation_epoch_end(self, outputs: List[List[Dict[str, List[str]]]]) -> None:
        def collect_loss(output: List[Dict[str, List[str]]]):
            return t.mean(t.Tensor([d["val_loss"] for d in output]))

        wer_clean = self._calculate_wer_on_collected_output(outputs[0])
        wer_other = self._calculate_wer_on_collected_output(outputs[1])

        self.log_dict(
            {
                "val_loss_clean": collect_loss(outputs[0]),
                "val_loss_other": collect_loss(outputs[1]),
                "val_wer_clean": wer_clean,
                "val_wer_other": wer_other,
            },
            prog_bar=True,
        )

    def test_step(
        self,
        batch: SpeechRecognitionDataBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        _, (letter_prediction, letter_prediction_lengths) = self.forward(
            batch.network_input, batch.input_lengths
        )

        with torch.no_grad():
            transcription = self._decode_letter_prediction(
                letter_prediction, letter_prediction_lengths
            )
            ground_truth_transcription = batch.ground_truth_strings

        return {
            "transcription": transcription,
            "ground_truth": ground_truth_transcription,
        }

    def test_epoch_end(self, outputs: List[List[Dict[str, List[str]]]]) -> None:
        wer_clean = self._calculate_wer_on_collected_output(outputs[0])
        wer_other = self._calculate_wer_on_collected_output(outputs[1])

        self.log_dict({"test_wer_clean": wer_clean, "test_wer_other": wer_other})

    @staticmethod
    def _calculate_wer_on_collected_output(output: List[Dict[str, List[str]]]):
        transcriptions = []
        ground_truths = []

        for d in output:
            transcriptions.extend(d["transcription"])
            ground_truths.extend(d["ground_truth"])

        return calculate_wer(transcriptions, ground_truths)

    def _decode_letter_prediction(
        self, letter_prediction: t.Tensor, lengths: List[int]
    ) -> List[str]:
        # letter prediction has shape [BATCH_SIZE, MAX_SEQUENCE_LENGTH
        batch_size = letter_prediction.shape[0]
        transcriptions = []

        for bs in range(0, batch_size):
            batch_seq = letter_prediction[bs, 0 : lengths[bs], :]

            highest_letter_idx = t.argmax(batch_seq, dim=1)

            transcription = self.tokenizer.decode_tensor(highest_letter_idx)
            transcriptions.append(transcription)

        return transcriptions

    def _log_transcription_progress(
        self, current_training_batch: SpeechRecognitionDataBatch
    ):
        if self.tracking_transcription is None:
            self.tracking_transcription = current_training_batch.ground_truth_strings[0]
            self.tracking_audio_sample = t.clone(
                current_training_batch.network_input[
                    0,
                ]
            ).detach()
            self.tracking_sequence_length = current_training_batch.input_lengths[0]

            if hasattr(self.logger, "experiment") and hasattr(
                self.logger.experiment, "log_text"
            ):
                self.logger.experiment.log_text(
                    f"ground_truth={self.tracking_transcription}",
                )
                self.logger.experiment.log_audio(
                    self.tracking_audio_sample.cpu().numpy(),
                    sample_rate=16000,
                    file_name="ground_truth.wav",
                )

        with torch.no_grad():
            _, (letter_prediction, letter_prediction_length) = self.forward(
                self.tracking_audio_sample, [self.tracking_sequence_length]
            )
            transcription = self._decode_letter_prediction(
                letter_prediction, letter_prediction_length
            )[0]

            if hasattr(self.logger, "experiment") and hasattr(
                self.logger.experiment, "log_text"
            ):
                self.logger.experiment.log_text(
                    f"`{transcription}` len={len(transcription)}", step=self.global_step
                )
