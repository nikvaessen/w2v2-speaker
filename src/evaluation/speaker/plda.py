################################################################################
#
# Implement the PLDA evaluation metric and evaluator.

# Author(s): Nik Vaessen
################################################################################
from collections import defaultdict
from typing import List, Tuple

import torch as t
import bob.learn.em as em
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.evaluation.speaker.speaker_recognition_evaluator import (
    EmbeddingSample,
    SpeakerRecognitionEvaluator,
    compute_mean_std_batch,
    center_batch,
    length_norm_batch,
)


################################################################################
# Implement an evaluator based PLDA scoring


class PLDAEvaluator(SpeakerRecognitionEvaluator):
    def __init__(
        self,
        num_lda_pca_components: int,
        num_plda_pca_components: int,
        max_iterations: int,
        max_training_batches_to_fit: int,
    ):
        super().__init__(max_training_batches_to_fit=max_training_batches_to_fit)

        self.num_lda_pca_components = num_lda_pca_components
        self.num_plda_pca_components = num_plda_pca_components
        self.max_iterations = max_iterations

        self._trainer = em.PLDATrainer()

        # set in self#fit_parameters
        self._lda_model: LinearDiscriminantAnalysis = None
        self._plda_model: em.PLDAMachine = None
        self._mean: t.Tensor = None
        self._std: t.Tensor = None

    def fit_parameters(
        self, embedding_tensors: List[t.Tensor], label_tensors: List[t.Tensor]
    ):
        # create a tensor of shape [BATCH_SIZE*len(tensors), EMBEDDING_SIZE]
        all_tensors = t.cat(embedding_tensors)
        all_labels = t.cat(label_tensors)

        # convert to numpy
        all_tensors = all_tensors.detach().cpu().numpy()
        all_labels = all_labels.detach().cpu().numpy().tolist()

        # fit LDA model and transform training data to
        # reduced LDA space
        # self._lda_model = LinearDiscriminantAnalysis(
        #     n_components=self.num_lda_pca_components
        # )
        # all_tensors_transformed = self._lda_model.fit_transform(all_tensors, all_labels)
        self._lda_model = PCA(n_components=self.num_lda_pca_components, whiten=True)
        all_tensors_transformed = self._lda_model.fit_transform(all_tensors, all_labels)

        # apply centering and length-norm in LDA space before training
        # PLDA
        all_tensors_transformed = t.Tensor(all_tensors_transformed)
        self._mean, self._std = compute_mean_std_batch(all_tensors_transformed)
        all_tensors_transformed = center_batch(
            all_tensors_transformed, self._mean, self._std
        )
        all_tensors_transformed = length_norm_batch(all_tensors_transformed)
        all_tensors_transformed = all_tensors_transformed.numpy()

        # Prepare for training PLDA model by converting to a list of tensors,
        # where each tensor is of shape [num_samples, num_embedding_size] and
        # all samples are from a single class
        class_map = defaultdict(list)

        for row_idx, class_label in enumerate(all_labels):
            class_map[class_label].append(np.squeeze(all_tensors_transformed[row_idx]))

        data_list = [
            np.stack(tensor_list)
            for tensor_list in class_map.values()
            if len(tensor_list) > 0
        ]

        # train the PLDA model
        # self._trainer.init_f_method = "BETWEEN_SCATTER"
        # self._trainer.init_g_method = "WITHIN_SCATTER"
        # self._trainer.init_sigma_method = "VARIANCE_DATA"

        plda_base = em.PLDABase(
            dim_d=self.num_lda_pca_components,  # Dimensionality of the feature vector.
            dim_f=self.num_plda_pca_components,  # Size of F (between class variantion matrix).
            dim_g=self.num_plda_pca_components,  # Size of G (within class variantion matrix).
            variance_threshold=1e-5,
        )
        em.train(
            self._trainer,
            plda_base,
            data_list,
            max_iterations=self.max_iterations,
            check_inputs=True,
        )
        self._plda_model = em.PLDAMachine(plda_base)

    def reset_parameters(self):
        self._mean = None
        self._std = None
        self._lda_model = None
        self._plda_model = None

    def _compute_prediction_scores(
        self, pairs: List[Tuple[EmbeddingSample, EmbeddingSample]]
    ) -> List[float]:
        # get 2 tensors of size [NUM_SAMPLES, EMBEDDING_SIZE],
        # where the same row idx corresponds to a pair to score
        b1, b2 = self._transform_pairs_to_tensor(pairs)

        # convert to latent dimension
        b1 = self._lda_model.transform(b1.detach().cpu().numpy())
        b2 = self._lda_model.transform(b2.detach().cpu().numpy())

        # center and length-norm in latent dimension
        b1 = t.Tensor(b1)
        b2 = t.tensor(b2)

        b1 = center_batch(b1, self._mean, self._std)
        b2 = center_batch(b2, self._mean, self._std)

        b1 = length_norm_batch(b1).numpy()
        b2 = length_norm_batch(b2).numpy()

        # compute a log-likelihood score for each pair
        score_list = []

        for idx in range(b1.shape[0]):
            embedding1 = np.squeeze(b1[idx])
            embedding2 = np.squeeze(b2[idx])

            pairing = np.stack((embedding1, embedding2))
            # debug_tensor_content(
            #     t.Tensor(pairing),
            #     "pairing",
            #     print_full_tensor=True,
            #     save_dir=pathlib.Path(
            #         "/home/nik/workspace/phd/repos/wav2vec_speaker_identification/models"
            #     ),
            # )
            print(pairing)
            print(pairing.shape)
            ll_ratio = self._plda_model.compute_log_likelihood(pairing)
            print(ll_ratio)
            print()
            score_list.append(10 ** ll_ratio)

        return score_list
