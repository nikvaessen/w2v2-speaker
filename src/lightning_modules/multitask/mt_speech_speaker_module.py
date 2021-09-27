################################################################################
#
# Base lightning module for multi-task learning of speech recognition
# and speaker recognition
#
# Author(s): Nik Vaessen
################################################################################

from abc import abstractmethod
from collections import deque
from typing import Optional, Callable, List, Tuple, Dict

import tqdm
import torch
import torchmetrics

import torch as t

from omegaconf import DictConfig

from src.data.modules.speaker.training_batch_speaker import (
    SpeakerClassificationDataBatch,
)
from src.data.modules.speech.training_batch_speech import SpeechRecognitionDataBatch
from src.evaluation.speaker.speaker_recognition_evaluator import (
    SpeakerRecognitionEvaluator,
    EvaluationPair,
    EmbeddingSample,
)
from src.evaluation.speech.wer import calculate_wer
from src.lightning_modules.base_lightning_module import BaseLightningModule
from src.tokenizer.base import BaseTokenizer

################################################################################
# lightning module


class SpeakerAndSpeechRecognitionModule(BaseLightningModule):
    def __init__(
        self,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        evaluator: SpeakerRecognitionEvaluator,
        num_speakers: int,
        embedding_size: int,
        validation_pairs: List[EvaluationPair],
        test_pairs: List[EvaluationPair],
        tokenizer: BaseTokenizer,
        auto_lr_find: Optional[float] = None,
    ):
        super().__init__(hyperparameter_config, loss_fn_constructor)

        # input arguments
        self.num_speakers = num_speakers
        self.embedding_size = embedding_size
        self.validation_pairs = validation_pairs
        self.test_pairs = test_pairs

        # used to keep track of training/val accuracy
        self.metric_train_loss = torchmetrics.AverageMeter()
        self.metric_train_loss_speech = torchmetrics.AverageMeter()
        self.metric_train_loss_speaker = torchmetrics.AverageMeter()
        self.metric_train_acc = torchmetrics.Accuracy()
        self.metric_train_wer = torchmetrics.AverageMeter()

        self.metric_valid_acc = torchmetrics.Accuracy()
        self.metric_val_loss_speech_clean = torchmetrics.AverageMeter()
        self.metric_val_loss_speech_other = torchmetrics.AverageMeter()

        # evaluator
        self.evaluator = evaluator

        # keep track of training embeddings for evaluator
        self.embeddings_queue = deque(maxlen=self.evaluator.max_num_training_samples)

        # set on first call to self#_log_transcription_progress step
        self.tracking_audio_sample: t.Tensor = None
        self.tracking_transcription: str = None
        self.tracking_sequence_length: int = None

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

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        pass

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        raise NotImplemented()

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        raise NotImplemented()

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
            tensor_list.append(embedding)
            label_list.append(label)

            if len(tensor_list) >= requested_amount:
                break

        return tensor_list, label_list

    @staticmethod
    def _extract_embedding_sample_list(outputs: List[dict]):
        embedding_list: List[EmbeddingSample] = []

        for d in outputs:
            embedding_tensor = d["embedding"]
            sample_id_list = d["sample_id"]

            if len(sample_id_list) != embedding_tensor.shape[0]:
                raise ValueError("batch dimension is missing or incorrect")

            for idx, sample_id in enumerate(sample_id_list):
                embedding_list.append(
                    EmbeddingSample(sample_id, embedding_tensor[idx, :].squeeze())
                )

        return embedding_list

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
