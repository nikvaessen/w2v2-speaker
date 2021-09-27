################################################################################
#
# Define a base lightning module for a speaker recognition network.
#
# Author(s): Nik Vaessen
################################################################################

import logging

from abc import abstractmethod
from typing import Callable, Optional, List, Any, Tuple
from collections import deque

import torch as t
import torch.nn

import torchmetrics
import tqdm

from pytorch_lightning.core.decorators import auto_move_data
from omegaconf import DictConfig

from src.callbacks.input_monitor_callback import debug_log_batch
from src.data.modules.speaker.training_batch_speaker import (
    SpeakerClassificationDataBatch,
)
from src.evaluation.speaker.speaker_recognition_evaluator import (
    EvaluationPair,
    SpeakerRecognitionEvaluator,
    EmbeddingSample,
)
from src.lightning_modules.base_lightning_module import BaseLightningModule
from src.optim.loss import (
    AngularAdditiveMarginSoftMaxLoss,
    CrossEntropyLoss,
    TripletLoss,
    TripletCrossEntropyLoss,
    CtcLoss,
)

################################################################################
# Definition of speaker recognition API

# A logger for this file

log = logging.getLogger(__name__)


class SpeakerRecognitionLightningModule(BaseLightningModule):
    def __init__(
        self,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        evaluator: SpeakerRecognitionEvaluator,
        num_speakers: int,
        embedding_size: int,
        validation_pairs: List[EvaluationPair],
        test_pairs: List[EvaluationPair],
        embeddings_are_pooled: bool,
        auto_lr_find: Optional[float] = None,
    ):
        super().__init__(hyperparameter_config, loss_fn_constructor, auto_lr_find)

        # input arguments
        self.num_speakers = num_speakers
        self.embedding_size = embedding_size
        self.validation_pairs = validation_pairs
        self.test_pairs = test_pairs

        # used to keep track of training/val accuracy
        self.metric_train_acc = torchmetrics.Accuracy()
        self.metric_train_loss = torchmetrics.AverageMeter()
        self.metric_valid_acc = torchmetrics.Accuracy()

        # evaluator
        self.evaluator = evaluator

        # keep track of training embeddings for evaluator
        self.embeddings_queue = deque(maxlen=self.evaluator.max_num_training_samples)

        # set exact behaviour of train/val/test step
        self.embeddings_are_pooled = embeddings_are_pooled
        self.training_mode = self._determine_training_mode()

    def _determine_training_mode(self):
        if isinstance(self.loss_fn, TripletCrossEntropyLoss):
            if not self.embeddings_are_pooled:
                raise ValueError("triplet loss does not support no_pooling")
            return "triplet_ce_loss"
        if isinstance(self.loss_fn, TripletLoss):
            if not self.embeddings_are_pooled:
                raise ValueError("triplet loss does not support no_pooling")
            return "triplet_loss"
        if isinstance(self.loss_fn, CrossEntropyLoss):
            if not self.embeddings_are_pooled:
                return "ce_loss_no_pool"
            else:
                return "ce_loss"
        if isinstance(self.loss_fn, AngularAdditiveMarginSoftMaxLoss):
            if not self.embeddings_are_pooled:
                return "ce_loss_no_pool"
            else:
                return "ce_loss"
        if isinstance(self.loss_fn, CtcLoss):
            # we need 1 more class index for blank
            self.num_speakers += 1
            return "ctc_loss"

    @abstractmethod
    def compute_speaker_embedding(self, input_tensor: t.Tensor) -> t.Tensor:
        # transform input_tensor with shape [BATCH_SIZE, ...]
        # into an embedding of shape [BATCH_SIZE, EMBEDDING_SIZE]
        pass

    @abstractmethod
    def compute_speaker_prediction(self, embedding_tensor: t.Tensor) -> t.Tensor:
        # transform embedding tensor with shape [BATCH_SIZE, EMBEDDING_SIZE]
        # into a speaker prediction of shape [BATCH_SIZE, NUM_SPEAKERS]
        pass

    @auto_move_data
    def forward(self, input_tensor: torch.Tensor):
        embedding = self.compute_speaker_embedding(input_tensor)

        if self.training_mode != "triplet_loss":
            prediction = self.compute_speaker_prediction(embedding)
        else:
            prediction = None

        return embedding, prediction

    def _add_batch_to_embedding_queue(self, embedding: t.Tensor, label: t.Tensor):
        # unbind embedding of shape [BATCH_SIZE, EMBEDDING_SIZE] into a list of
        # tensors of shape [EMBEDDING_SIZE] with len=BATCH_SIZE
        embedding_list = t.unbind(embedding, dim=0)

        # unbind labels of shape [BATCH_SIZE] into a list of tensors of shape [1]
        # with len=BATCH_SIZE
        label_list = t.unbind(label, dim=0)

        # add embedding and label to queue
        assert len(embedding_list) == len(label_list)

        self.embeddings_queue.extend(
            [(embedding, label) for embedding, label in zip(embedding_list, label_list)]
        )

    def training_step(
        self,
        batch: SpeakerClassificationDataBatch,
        batch_idx: int,
        optimized_idx: Optional[int] = None,
    ):
        audio_input = batch.network_input
        label = batch.ground_truth

        try:
            if self.training_mode == "ce_loss":
                loss, embedding, label = self._train_step_ce_loss(
                    audio_input, label, batch_idx
                )
            elif self.training_mode == "ce_loss_no_pool":
                loss, embedding, label = self._train_step_ce_loss_no_pooling(
                    audio_input, label, batch_idx
                )
            elif self.training_mode == "triplet_loss":
                loss, embedding, label = self._train_step_triplet_loss(
                    audio_input, label, batch_idx
                )
            elif self.training_mode == "triplet_ce_loss":
                loss, embedding, label = self._train_step_triplet_ce_loss(
                    audio_input, label, batch_idx
                )
            elif self.training_mode == "ctc_loss":
                loss, embedding, label = self._train_step_ctc_loss(
                    audio_input, label, batch_idx
                )
            else:
                raise ValueError(f"unrecognised training mode {self.training_mode}")

            self._log_train_loss(loss, batch_idx)

            with t.no_grad():
                self._add_batch_to_embedding_queue(
                    embedding.detach().to("cpu"), label.detach().to("cpu")
                )
        except Exception as e:
            additional_tensors = {}

            try:
                additional_tensors["embedding"] = embedding
                additional_tensors["loss"] = loss
            except NameError:
                pass

            debug_log_batch(
                batch,
                name="train_step",
                additional_tensors=additional_tensors,
                write_whole_tensor_to_file=False,
            )

            raise e

        return {"loss": loss}

    def _train_step_ce_loss(
        self, audio_input: t.Tensor, spk_label: t.Tensor, batch_idx: int
    ):
        embedding = self.compute_speaker_embedding(audio_input)

        assert len(embedding.shape) == 2
        assert embedding.shape[-1] == self.embedding_size

        logits_prediction = self.compute_speaker_prediction(embedding)
        loss, prediction = self.loss_fn(logits_prediction, spk_label)

        self._log_train_acc(prediction, spk_label, batch_idx)

        return loss, embedding, spk_label

    def _train_step_ctc_loss(
        self, audio_input: t.Tensor, spk_label: t.Tensor, batch_idx: int
    ):
        embedding = self.compute_speaker_embedding(audio_input)

        # add +1 to spk labels to prevent '0' label (which is blank symbol)
        spk_label += 1

        assert len(embedding.shape) == 3
        assert embedding.shape[-1] == self.embedding_size

        logits_prediction = self.compute_speaker_prediction(embedding)

        batch_size = embedding.shape[0]
        pred_length = t.ones((batch_size)) * logits_prediction.shape[1]
        pred_length = pred_length.to(t.int32)

        gt_length = t.ones((batch_size))
        gt_length = gt_length.to(t.int32)

        loss = self.loss_fn(logits_prediction, pred_length, spk_label, gt_length)

        return loss, embedding, spk_label

    def _train_step_ce_loss_no_pooling(
        self, audio_input: t.Tensor, spk_label: t.Tensor, batch_idx: int
    ):
        embedding = self.compute_speaker_embedding(audio_input)

        assert len(embedding.shape) == 3
        assert embedding.shape[-1] == self.embedding_size

        logits_prediction = self.compute_speaker_prediction(embedding)

        # if we didn't use any pooling each embedding has it's own prediction
        # so we need to flatten out the batch and repeat the labels
        # for each time step
        flat_spk_label = t.repeat_interleave(spk_label, embedding.shape[1])
        flat_logits_prediction = t.flatten(logits_prediction, start_dim=0, end_dim=1)
        flat_embedding = t.flatten(embedding, start_dim=0, end_dim=1)

        loss, prediction = self.loss_fn(flat_logits_prediction, flat_spk_label)

        self._log_train_acc(prediction, flat_spk_label, batch_idx)

        return loss, flat_embedding, flat_spk_label

    def _train_step_triplet_loss(
        self, audio_input: t.Tensor, spk_label: t.Tensor, batch_idx: int
    ):
        embedding = self.compute_speaker_embedding(audio_input)

        assert len(embedding.shape) == 2
        assert embedding.shape[-1] == self.embedding_size

        loss = self.loss_fn(embedding, spk_label)

        return loss, embedding, spk_label

    def _train_step_triplet_ce_loss(
        self, audio_input: t.Tensor, spk_label: t.Tensor, batch_idx: int
    ):
        embedding = self.compute_speaker_embedding(audio_input)

        assert len(embedding.shape) == 2
        assert embedding.shape[-1] == self.embedding_size

        logits_prediction = self.compute_speaker_prediction(embedding)
        loss, prediction = self.loss_fn(embedding, logits_prediction, spk_label)

        self._log_train_acc(prediction, spk_label, batch_idx)

        return loss, embedding, spk_label

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
        batch: SpeakerClassificationDataBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ):
        audio_input = batch.network_input
        label = batch.ground_truth
        sample_id = batch.keys

        if self.training_mode == "ce_loss":
            embedding = self._val_step_ce_loss(audio_input, label, batch_idx)
        elif self.training_mode == "ce_loss_no_pool":
            embedding = self._val_step_ce_loss_no_pooling(audio_input, label, batch_idx)
        elif self.training_mode == "triplet_loss":
            embedding = self._val_step_triplet_loss(audio_input, label, batch_idx)
        elif self.training_mode == "triplet_ce_loss":
            embedding = self._val_step_triplet_ce_loss(audio_input, label, batch_idx)
        elif self.training_mode == "ctc_loss":
            embedding = self._val_step_ctc_loss(audio_input, label, batch_idx)
        else:
            raise ValueError(f"unrecognised training mode {self.training_mode}")

        return {"embedding": embedding.detach().to("cpu"), "sample_id": sample_id}

    def _val_step_ce_loss(self, audio_input: t.Tensor, label: t.Tensor, batch_idx: int):
        embedding = self.compute_speaker_embedding(audio_input)

        assert len(embedding.shape) == 2
        assert embedding.shape[-1] == self.embedding_size

        logits_prediction = self.compute_speaker_prediction(embedding)
        loss, prediction = self.loss_fn(logits_prediction, label)

        self.metric_valid_acc(prediction, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return embedding

    def _val_step_ctc_loss(
        self, audio_input: t.Tensor, label: t.Tensor, batch_idx: int
    ):
        self.training = True  # force `none` pooling
        embedding = self.compute_speaker_embedding(audio_input)

        # add +1 to spk labels to prevent '0' label (which is blank symbol)
        label += 1

        print(f"val embedding: {embedding.shape} {self.training}")

        assert len(embedding.shape) == 3
        assert embedding.shape[-1] == self.embedding_size

        logits_prediction = self.compute_speaker_prediction(embedding)

        batch_size = embedding.shape[0]
        pred_length = t.ones((batch_size)) * logits_prediction.shape[1]
        pred_length = pred_length.to(t.int32)

        gt_length = t.ones((batch_size))
        gt_length = gt_length.to(t.int32)

        loss = self.loss_fn(logits_prediction, pred_length, label, gt_length)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # manually pool
        embedding = self.stat_pooling(embedding)

        return embedding

    def _val_step_ce_loss_no_pooling(
        self, audio_input: t.Tensor, label: t.Tensor, batch_idx: int
    ):
        embedding = self.compute_speaker_embedding(audio_input)

        assert len(embedding.shape) == 3
        assert embedding.shape[-1] == self.embedding_size

        logits_prediction = self.compute_speaker_prediction(embedding)

        # if we didn't use any pooling each embedding has it's own prediction
        # so we need to flatten out the batch and repeat the labels
        # for each time step
        flat_spk_label = t.repeat_interleave(label, embedding.shape[1])
        flat_logits_prediction = t.flatten(logits_prediction, start_dim=0, end_dim=1)
        flat_embedding = t.flatten(embedding, start_dim=0, end_dim=1)

        loss, prediction = self.loss_fn(flat_logits_prediction, flat_spk_label)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.metric_valid_acc(prediction, flat_spk_label)

        return loss, flat_embedding

    def _val_step_triplet_loss(
        self, audio_input: t.Tensor, label: t.Tensor, batch_idx: int
    ):
        embedding = self.compute_speaker_embedding(audio_input)
        assert len(embedding.shape) == 2
        assert embedding.shape[-1] == self.embedding_size

        try:
            loss = self.loss_fn(embedding, label)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        except:
            # no triplets in remaining batches
            pass

        return embedding

    def _val_step_triplet_ce_loss(
        self, audio_input: t.Tensor, label: t.Tensor, batch_idx: int
    ):
        embedding = self.compute_speaker_embedding(audio_input)
        assert len(embedding.shape) == 2
        assert embedding.shape[-1] == self.embedding_size

        logits_prediction = self.compute_speaker_prediction(embedding)

        try:
            loss, prediction = self.loss_fn(embedding, logits_prediction, label)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.metric_valid_acc(prediction, label)
        except:
            # no triplets in remaining batches
            pass

        return embedding

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # queue is only ever empty here in the 'sanity_check' val loop
        # which doesn't need EER results anyway.
        if len(self.embeddings_queue) >= self.evaluator.max_num_training_samples:
            results = self._evaluate_embeddings(outputs, self.validation_pairs)
            self.log("val_eer", results["eer"], prog_bar=True)

        if self.training_mode not in ["triplet_loss", "ctc_loss"]:
            self.log("val_acc", self.metric_valid_acc.compute(), prog_bar=True)
            self.metric_valid_acc.reset()

    def test_step(
        self,
        batch: SpeakerClassificationDataBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ):
        if batch.batch_size != 1:
            raise ValueError("expecting a batch size of 1 for evaluation")

        audio_input = batch.network_input
        sample_id = batch.keys

        if (
            hasattr(self, "compute_ensemble_embedding")
            and hasattr(self, "test_with_ensemble")
            and self.test_with_ensemble
        ):
            embedding = self.compute_ensemble_embedding(audio_input)
            embedding = [t.stack([e.detach().to("cpu")]) for e in embedding]
            for e in embedding:
                assert len(e.shape) == 2
                assert e.shape[0] == batch.batch_size
                assert e.shape[0] == self.embedding_size
        else:
            embedding = self.compute_speaker_embedding(audio_input)

            if len(embedding.shape) == 1 and embedding.shape[0] == self.embedding_size:
                embedding = embedding[None, :]

            assert len(embedding.shape) == 2
            assert embedding.shape[0] == batch.batch_size
            assert embedding.shape[1] == self.embedding_size

            embedding = t.stack([embedding.detach().to("cpu")])

        return {
            "embedding": embedding,
            "sample_id": sample_id,
        }

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self._evaluate_embeddings(outputs, self.test_pairs))

    def _evaluate_embeddings(self, outputs: List[dict], pairs: List[EvaluationPair]):
        # outputs is a list of dictionary with at least keys:
        # 'embedding' -> tensor with shape [BATCH_SIZE, EMBEDDING_SIZE]
        # 'sample_id' -> list of keys with length BATCH_SIZE
        embedding_list = self._extract_embedding_sample_list(outputs)

        training_embeddings, training_labels = self._collect_training_embeddings(
            self.evaluator.max_num_training_samples
        )

        self.evaluator.reset_parameters()
        self.evaluator.fit_parameters(training_embeddings, training_labels)
        result = self.evaluator.evaluate(pairs, embedding_list)

        return result

    def _collect_training_embeddings(
        self, requested_amount: int = None
    ) -> Tuple[List[t.Tensor], List[t.Tensor]]:
        if len(self.embeddings_queue) < requested_amount:
            with t.no_grad():
                # collect the requested amount of batches
                with tqdm.tqdm(
                    total=requested_amount - len(self.embeddings_queue)
                ) as p:
                    p.write("loading training embeddings to fit evaluator on")

                    for batch in self.train_dataloader():
                        if (
                            len(self.embeddings_queue) >= requested_amount
                            or len(self.embeddings_queue)
                            >= self.embeddings_queue.maxlen
                        ):
                            break

                        batch: SpeakerClassificationDataBatch = batch

                        embedding, prediction = self(batch.network_input)
                        self._add_batch_to_embedding_queue(
                            embedding, batch.ground_truth
                        )

                        p.update(batch.batch_size)

        # return a list of tensors with shape [EMBEDDING_SIZE]
        # length of list is min(requested_amount, len(training_dataloader)
        tensor_list = []
        label_list = []

        for embedding, label in self.embeddings_queue:
            tensor_list.append(embedding.to("cpu"))
            label_list.append(label.to("cpu"))

            if len(tensor_list) >= requested_amount:
                break

        return tensor_list, label_list

    @staticmethod
    def _extract_embedding_sample_list(outputs: List[dict]):
        embedding_list: List[EmbeddingSample] = []

        for d in outputs:
            embedding_tensor = d["embedding"]
            sample_id_list = d["sample_id"]

            if isinstance(embedding_tensor, list):
                if len(sample_id_list) != embedding_tensor[0].shape[0]:
                    raise ValueError("batch dimension is missing or incorrect")
            else:
                if len(sample_id_list) != embedding_tensor.shape[0]:
                    raise ValueError("batch dimension is missing or incorrect")

            for idx, sample_id in enumerate(sample_id_list):
                if isinstance(embedding_tensor, list):
                    embedding_list.append(
                        EmbeddingSample(
                            sample_id=sample_id,
                            embedding=[e[idx, :].squeeze() for e in embedding_tensor],
                        )
                    )
                else:
                    embedding_list.append(
                        EmbeddingSample(
                            sample_id=sample_id,
                            embedding=embedding_tensor[idx, :].squeeze(),
                        )
                    )

        return embedding_list
