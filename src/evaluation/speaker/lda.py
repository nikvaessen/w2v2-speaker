################################################################################
#
# Implement the PLDA evaluation metric and evaluator.

# Author(s): Nik Vaessen
################################################################################

from typing import List, Tuple

import torch as t
from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.evaluation.speaker.cosine_distance import (
    compute_cosine_scores,
)
from src.evaluation.speaker.speaker_recognition_evaluator import (
    EmbeddingSample,
    compute_mean_std_batch,
    center_batch,
    SpeakerRecognitionEvaluator,
    length_norm_batch,
)


################################################################################
# Implement an evaluator based PLDA scoring


class LDAEvaluator(SpeakerRecognitionEvaluator):
    def __init__(
        self,
        center_before_scoring: bool,
        length_norm_before_scoring: bool,
        max_training_batches_to_fit: int,
        num_pca_components: int,
        center_before_fit_training_batches: bool,
    ):
        super().__init__(
            max_training_batches_to_fit=max_training_batches_to_fit,
        )

        self.center_before_scoring = center_before_scoring
        self.length_norm_before_scoring = length_norm_before_scoring
        self.num_pca_components = num_pca_components
        self.center_before_fit_training_batches = center_before_fit_training_batches

        # set in self#fit_parameters
        self._lda_model: LinearDiscriminantAnalysis = None
        self._mean: t.Tensor = None
        self._std: t.Tensor = None

    def fit_parameters(
        self, embedding_tensors: List[t.Tensor], label_tensors: List[t.Tensor]
    ):
        # create a tensor of shape [BATCH_SIZE*len(embedding_tensors), EMBEDDING_SIZE]
        all_tensors = t.cat(embedding_tensors)

        # create a tensor of SHAPE [BATCH_SIZE*len(label_tensors),]
        all_labels = t.cat(label_tensors)

        if self.center_before_fit_training_batches:
            mean, std = compute_mean_std_batch(all_tensors)
            all_tensors = center_batch(all_tensors, mean, std)

        # convert to numpy
        all_tensors = all_tensors.detach().cpu().numpy()
        all_labels = all_labels.detach().cpu().numpy().tolist()

        # train LDA model
        self._lda_model = PCA(n_components=200, whiten=True)
        all_tensors_transformed = self._lda_model.fit_transform(all_tensors, all_labels)

        # compute mean/std in latent space in order to do centering before
        # taking length norm
        self._mean, self._std = compute_mean_std_batch(
            t.Tensor(all_tensors_transformed)
        )

    def reset_parameters(self):
        super().reset_parameters()
        self._lda_model = None

    def _compute_prediction_scores(
        self, pairs: List[Tuple[EmbeddingSample, EmbeddingSample]]
    ) -> List[float]:
        # get 2 tensors of size [NUM_SAMPLES, EMBEDDING_SIZE],
        # where the same row idx corresponds to a pair to score
        b1, b2 = self._transform_pairs_to_tensor(pairs)

        # convert to latent dimension
        b1 = self._lda_model.transform(b1.detach().cpu().numpy())
        b2 = self._lda_model.transform(b2.detach().cpu().numpy())

        # convert back to tensors
        b1 = t.Tensor(b1)
        b2 = t.Tensor(b2)

        if self.center_before_scoring:
            b1 = center_batch(b1, self._mean, self._std)
            b2 = center_batch(b2, self._mean, self._std)

        if self.length_norm_before_scoring:
            b1 = length_norm_batch(b1)
            b2 = length_norm_batch(b2)

        # compute scores based on centering, length norming and then
        # taking cosine distance
        return compute_cosine_scores(t.Tensor(b1), t.Tensor(b2))
