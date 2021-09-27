################################################################################
#
# This run script encapsulates the training and evaluation of a module
# defined by the hydra configuration.
#
# Author(s): Nik Vaessen
################################################################################

import logging
import datetime
import pathlib

from typing import List, Tuple, Union, Callable

import comet_ml
import hydra
import pytorch_lightning as pl

import torch
import torchaudio
import tqdm

import numpy as np
import torch as t

from omegaconf import DictConfig, OmegaConf
from pl_bolts.callbacks.verification.batch_gradient import BatchGradientVerification
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from hydra.utils import instantiate
from pytorch_model_summary import summary

from src.callbacks.memory_monitor import RamMemoryMonitor
from src.data.modules.speaker.speaker_data_module import SpeakerLightningDataModule
from src.data.modules.speech.librispeech import (
    LibriSpeechLightningDataModuleConfig,
    LibriSpeechLightningDataModule,
)
from src.data.modules.speaker.voxceleb import (
    VoxCelebDataModuleConfig,
    VoxCelebDataModule,
)
from src.data.common import (
    WebDataSetShardConfig,
    SpeakerDataLoaderConfig,
    SpeechDataLoaderConfig,
)
from src.data.modules.speech.speech_data_module import SpeechLightningDataModule
from src.data.preprocess.input_normalisation import InputNormalizer2D
from src.evaluation.speaker.speaker_recognition_evaluator import (
    EmbeddingSample,
    SpeakerRecognitionEvaluator,
)
from src.lightning_modules.speaker import (
    SpeakerRecognitionLightningModule,
    XVectorModuleConfig,
    XVectorModule,
    Wav2vecXVectorModuleConfig,
    Wav2vecXVectorModule,
    Wav2vecFCModuleConfig,
    Wav2vecFCModule,
    Wav2SpkModuleConfig,
    Wav2SpkModule,
    Wav2vec2FCModuleConfig,
    Wav2vec2FCModule,
    DummyModuleConfig,
    DummyModule,
    Wav2vec2PairedSpeakerModuleConfig,
    Wav2vec2PairedSpeakerModule,
    PairedSpeakerRecognitionLightningModule,
    EcapaTDNNModuleConfig,
    EcapaTdnnModule,
)
from src.lightning_modules.speech.speech_recognition_module import (
    SpeechRecognitionLightningModule,
)
from src.lightning_modules.speech.wav2vec2_fc_letter import (
    Wav2vec2FcLetterRecognizerConfig,
    Wav2vec2FcLetterRecognizer,
)
from src.optim.loss import AngularAdditiveMarginSoftMaxLoss
from src.tokenizer.tokenizer_wav2vec2 import Wav2vec2TokenizerConfig, Wav2vec2Tokenizer
from src.main import construct_data_module, construct_module

################################################################################
# entrypoint of hydra script doing prediction on (unlabeled) data


def run_predictions(cfg: DictConfig):
    # create data module
    dm = construct_data_module(cfg)

    # create evaluator (for speaker recognition)
    evaluator: SpeakerRecognitionEvaluator = instantiate(cfg.evaluator)

    # create network module
    module = construct_module(cfg, evaluator, dm, load_optim=False)
    module = module.eval()

    # load files and pairs
    folder_path = pathlib.Path(cfg.get("predict_folder_path"))
    pair_file = pathlib.Path(cfg.get("pair_prediction_path"))

    pairs: List[Tuple[str, str]] = []
    id_set = set()

    with open(pair_file, "r") as f:
        pair_names = [l.strip().split(" ") for l in f.readlines() if l.count(" ") > 0]

        for pair_tuple in pair_names:
            if len(pair_tuple) == 3:
                p1_name = pair_tuple[1]
                p2_name = pair_tuple[2]
            else:
                p1_name = pair_tuple[0]
                p2_name = pair_tuple[1]

            id_set.add(p1_name)
            id_set.add(p2_name)

            pairs.append((p1_name, p2_name))

    # make embeddings
    embedding_folder = folder_path / "embeddings"
    embedding_folder.mkdir(exist_ok=True)

    print("computing speaker embeddings")
    max_len = -1
    norm = InputNormalizer2D()

    for name in tqdm.tqdm(id_set):
        name: str = name
        save_path = embedding_folder / (name + ".pt")

        if save_path.exists():
            continue

        # load audio
        audio_tensor, sr = torchaudio.load(str(folder_path / name))
        audio_tensor, _, _ = norm.normalize(audio_tensor, False)

        audio_len = audio_tensor.shape[1]
        if audio_len > max_len:
            max_len = audio_len
            print(max_len)

        if sr != 16000:
            raise ValueError("expected sr 16000")

        # compute speaker embedding
        with torch.no_grad():
            try:
                audio_tensor = audio_tensor.to("cuda")
                module = module.to("cuda")

                embedding, _ = module(audio_tensor)
                embedding = embedding.to("cpu")
            except:
                # Just use cpu if input it too large for VRAM of gpu
                module = module.to("cpu")
                audio_tensor = audio_tensor.to("cpu")

                embedding, _ = module(audio_tensor)

        save_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(embedding, save_path)

        del embedding
        del audio_tensor

    # make and store pair predictions
    print("comparing pairs of speaker embeddings")

    embedding_pairs: List[Tuple[EmbeddingSample, EmbeddingSample]] = []
    for pair in tqdm.tqdm(pairs):
        p1_name: str = pair[0]
        p2_name: str = pair[1]

        p1_emb_path = embedding_folder / (p1_name + ".pt")
        p2_emb_path = embedding_folder / (p2_name + ".pt")

        if not p1_emb_path.exists():
            raise ValueError(f"{p1_emb_path} does not exist")

        if not p2_emb_path.exists():
            raise ValueError(f"{p2_emb_path} does not exist")

        p1 = EmbeddingSample(sample_id=p1_name, embedding=t.load(p1_emb_path))
        p2 = EmbeddingSample(sample_id=p2_name, embedding=t.load(p2_emb_path))

        embedding_pairs.append((p1, p2))

    print("computing cosine distance scores")
    scores = evaluator._compute_prediction_scores(embedding_pairs)

    # ensure scores are between 0 and 1
    scores = np.array(scores)
    scores = (scores + 1) / 2
    scores = np.clip(scores, 0, 1)
    scores = scores.tolist()

    # save predictions
    assert len(scores) == len(embedding_pairs)

    score_file = pair_file.parent / (pair_file.stem + "_scores.txt")
    print(f"writing scores to {score_file}")
    with score_file.open("w") as f:
        for i in range(len(scores)):
            score = scores[i]
            pair = embedding_pairs[i]
            file1 = pair[0].sample_id
            file2 = pair[1].sample_id

            line = f"{score} {file1} {file2}\n"
            f.write(line)
