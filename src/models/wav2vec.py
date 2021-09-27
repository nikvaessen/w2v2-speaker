################################################################################
#
# Provide embeddings from raw audio with the wav2vec model from fairseq.
#
# See `download/download_pretrained_models.sh` for links to pretrained weights.
#
# Author(s): Nik Vaessen
################################################################################

import pathlib

import fairseq
import pytorch_lightning
import torch as t

from fairseq.models.wav2vec import Wav2VecModel

from src.util import reset_model

################################################################################
# loading wav2vec with fairseq


def load_wav2vec_model(
    model_path: pathlib.Path, device: t.cuda.Device = t.device("cpu")
) -> Wav2VecModel:
    """
    Load the wav2vec model.

    :param model_path: path to the ".pt" file of the model
    :param device: the device on which the model should be loaded
    :return: the wav2vec2 model on the specified device
    """
    checkpoint = t.load(model_path)

    model = fairseq.models.wav2vec.Wav2VecModel.build_model(checkpoint["args"], None)
    model.load_state_dict(checkpoint["model"])

    return model.to(device)


################################################################################
# computation of embedding


def wav2vec_embed_raw_audio(
    input_tensor: t.Tensor, model: Wav2VecModel, aggregate: bool = False
) -> t.Tensor:
    """
    Calculate a [1, 512, num_frames] embedding of a given [1, num_samples] audio file
    by using the Wav2Vec model.

    :param input_tensor: a raw audio input (between -1 and 1) with a sampling rate of 16000 Hz
    :param model: the wav2vec model
    :param aggregate whether to apply an aggregation to the initial features
    :return: The embedding with shape [1, 512, num_frames], where num_frames < num_samples.
    """
    z = model.feature_extractor(input_tensor)

    if not aggregate:
        return z
    else:
        return model.feature_aggregator(z)


################################################################################
# wrap the wav2vec model


class Wav2VecWrapperModule(pytorch_lightning.LightningModule):
    def __init__(
        self,
        wav2vec_model_path: pathlib.Path,
        wav2vec_aggregation: bool = False,
        reset_weights: bool = False,
    ):
        super().__init__()

        self.model = load_wav2vec_model(wav2vec_model_path)
        self.use_aggregator = wav2vec_aggregation
        self.num_features = 512

        if reset_weights:
            reset_model(self.model)

    @property
    def num_embedding_features(self):
        return self.num_features

    def forward(self, wav_input: t.Tensor):
        # wav_input has shape [BATCH_SIZE, NUM_SAMPLES]
        embedding = wav2vec_embed_raw_audio(wav_input, self.model, self.use_aggregator)

        # return an embedding with shape [BATCH_SIZE, NUM_FEATURES, NUM_FRAMES]
        return embedding
