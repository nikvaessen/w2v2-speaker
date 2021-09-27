################################################################################
#
# This run script encapsulates the training and evaluation of a module
# defined by the hydra configuration.
#
# Author(s): Nik Vaessen
################################################################################
import json
import logging
import datetime
import pathlib

from typing import List, Tuple, Union, Callable

import comet_ml
import hydra
import pytorch_lightning as pl
import pytorch_lightning

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


################################################################################
# execute train and evaluation procedure as a hydra application

# A logger for this file

log = logging.getLogger(__name__)


def construct_data_module(
    cfg: DictConfig,
) -> Union[SpeechLightningDataModule, SpeakerLightningDataModule]:
    # load shard config
    shard_cfg: WebDataSetShardConfig = instantiate(cfg.data.shards)

    # load dataloader config
    dl_cfg = instantiate(cfg.data.dataloader)

    # load data pipelines
    if cfg.data.pipeline.get("augmentations", None) is not None:
        augment_wrappers = [
            instantiate(cfg.data.pipeline[n])
            for n in cfg.data.pipeline.get("augmentations")
        ]
    else:
        augment_wrappers = None

    train_pipeline = [
        instantiate(cfg.data.pipeline[n])
        if cfg.data.pipeline[n]["_target_"] != "src.data.preprocess.augment.Augmenter"
        else instantiate(cfg.data.pipeline[n], augmenters=augment_wrappers)
        for n in cfg.data.pipeline.train_pipeline
    ]
    val_pipeline = [
        instantiate(cfg.data.pipeline[n]) for n in cfg.data.pipeline.val_pipeline
    ]
    test_pipeline = [
        instantiate(cfg.data.pipeline[n]) for n in cfg.data.pipeline.test_pipeline
    ]

    # load data module config
    dm_cfg = hydra.utils.instantiate(cfg.data.module)

    # create data module
    dm: Union[SpeechLightningDataModule, SpeakerLightningDataModule] = None

    if isinstance(dm_cfg, VoxCelebDataModuleConfig):
        if not isinstance(dl_cfg, SpeakerDataLoaderConfig):
            raise ValueError(
                f"VoxCelebDataModule expects {SpeakerDataLoaderConfig},"
                f" got {dl_cfg}"
            )
        dm = VoxCelebDataModule(
            cfg=dm_cfg,
            shard_cfg=shard_cfg,
            dl_cfg=dl_cfg,
            train_pipeline=train_pipeline,
            val_pipeline=val_pipeline,
            test_pipeline=test_pipeline,
        )
    elif isinstance(dm_cfg, LibriSpeechLightningDataModuleConfig):
        if not isinstance(dl_cfg, SpeechDataLoaderConfig):
            raise ValueError(
                f"LibriSpeechLightningDataModule expects {SpeechDataLoaderConfig},"
                f" got {dl_cfg}"
            )

        dm = LibriSpeechLightningDataModule(
            cfg=dm_cfg,
            shard_cfg=shard_cfg,
            dl_cfg=dl_cfg,
            tokenizer=construct_tokenizer(cfg),
            train_pipeline=train_pipeline,
            val_pipeline=val_pipeline,
            test_pipeline=test_pipeline,
        )
    else:
        raise ValueError(f"cannot load data module from {dm_cfg}")

    dm.prepare_data()
    dm.setup()
    dm.summary()

    return dm


def construct_tokenizer(cfg: DictConfig):
    tokenizer_cfg = instantiate(cfg.tokenizer)

    if isinstance(tokenizer_cfg, Wav2vec2TokenizerConfig):
        tokenizer = Wav2vec2Tokenizer(tokenizer_cfg)
    else:
        raise ValueError(f"cannot construct a tokenizer based on {tokenizer_cfg}")

    return tokenizer


def construct_speech_recognition_module(
    cfg: DictConfig,
    network_cfg: DictConfig,
    dm: SpeechLightningDataModule,
    loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
) -> SpeechRecognitionLightningModule:
    # every network needs these variables
    tokenizer = dm.tokenizer()

    if isinstance(network_cfg, Wav2vec2FcLetterRecognizerConfig):
        network_class = Wav2vec2FcLetterRecognizer
    else:
        raise ValueError(f"cannot load network from {network_cfg}")

    # load model weights from checkpoint
    potential_checkpoint_path = cfg.get("load_network_from_checkpoint", None)

    if potential_checkpoint_path is not None:
        log.info(
            f"reloading {network_class.__class__} from {potential_checkpoint_path}"
        )
        network = network_class.load_from_checkpoint(
            cfg.load_network_from_checkpoint,
            hyperparameter_config=cfg,
            cfg=network_cfg,
            loss_fn_constructor=loss_fn_constructor,
            tokenizer=tokenizer,
            strict=False,
        )
    else:
        network = network_class(
            hyperparameter_config=cfg,
            cfg=network_cfg,
            loss_fn_constructor=loss_fn_constructor,
            tokenizer=tokenizer,
        )

    return network


def construct_speaker_recognition_module(
    cfg: DictConfig,
    network_cfg: DictConfig,
    evaluator: SpeakerRecognitionEvaluator,
    dm: SpeakerLightningDataModule,
    loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
) -> SpeakerRecognitionLightningModule:
    # every network needs to be given these variables
    num_speakers = dm.num_speakers
    validation_pairs = dm.val_pairs
    test_pairs = dm.test_pairs

    # get init function based on config type
    if isinstance(network_cfg, XVectorModuleConfig):
        network_class = XVectorModule
    elif isinstance(network_cfg, EcapaTDNNModuleConfig):
        network_class = EcapaTdnnModule
    elif isinstance(network_cfg, Wav2vecXVectorModuleConfig):
        network_class = Wav2vecXVectorModule
    elif isinstance(network_cfg, Wav2vecFCModuleConfig):
        network_class = Wav2vecFCModule
    elif isinstance(network_cfg, Wav2SpkModuleConfig):
        network_class = Wav2SpkModule
    elif isinstance(network_cfg, Wav2vec2FCModuleConfig):
        network_class = Wav2vec2FCModule
    elif isinstance(network_cfg, DummyModuleConfig):
        network_class = DummyModule
    elif isinstance(network_cfg, Wav2vec2PairedSpeakerModuleConfig):
        network_class = Wav2vec2PairedSpeakerModule
    else:
        raise ValueError(f"cannot load network from {network_cfg}")

    if issubclass(network_class, PairedSpeakerRecognitionLightningModule):
        kwargs = {
            "hyperparameters_to_save": cfg,
            "cfg": network_cfg,
            "loss_fn_constructor": loss_fn_constructor,
        }
    else:
        kwargs = {
            "hyperparameters_to_save": cfg,
            "cfg": network_cfg,
            "num_speakers": num_speakers,
            "loss_fn_constructor": loss_fn_constructor,
            "validation_pairs": validation_pairs,
            "test_pairs": test_pairs,
            "evaluator": evaluator,
        }

    # load model weights from checkpoint
    potential_checkpoint_path = cfg.get("load_network_from_checkpoint", None)

    if potential_checkpoint_path is not None:
        log.info(
            f"reloading {network_class.__class__} from {potential_checkpoint_path}"
        )
        network = network_class.load_from_checkpoint(
            cfg.load_network_from_checkpoint, strict=False, **kwargs
        )
    else:
        network = network_class(**kwargs)

    return network


def construct_module(
    cfg: DictConfig,
    evaluator: SpeakerRecognitionEvaluator,
    dm: Union[SpeakerLightningDataModule, SpeechLightningDataModule],
    load_optim: bool = True,
) -> Union[SpeakerRecognitionLightningModule, SpeechLightningDataModule]:
    # load loss function
    def loss_fn_constructor():
        # should be instantiated in the network
        # so that potential parameters are properly
        # registered
        return instantiate(cfg.optim.loss)

    # load network config
    network_cfg = instantiate(cfg.network)

    if isinstance(dm, SpeakerLightningDataModule) and isinstance(
        dm, SpeechLightningDataModule
    ):
        raise ValueError("No multi-task networks yet!")
    elif isinstance(dm, SpeakerLightningDataModule):
        network = construct_speaker_recognition_module(
            cfg, network_cfg, evaluator, dm, loss_fn_constructor
        )
    elif isinstance(dm, SpeechLightningDataModule):
        network = construct_speech_recognition_module(
            cfg, network_cfg, dm, loss_fn_constructor
        )
    else:
        raise ValueError(
            f"can not construct network for data module type {dm.__class__}"
        )

    # set optimizer and learning rate schedule
    if load_optim:
        optimizer = instantiate(cfg.optim.algo, params=network.parameters())
        schedule = {
            "scheduler": instantiate(cfg.optim.schedule.scheduler, optimizer=optimizer),
            "monitor": cfg.optim.schedule.monitor,
            "interval": cfg.optim.schedule.interval,
            "frequency": cfg.optim.schedule.frequency,
            "name": cfg.optim.schedule.name,
        }
        # remove None values from dict
        schedule = {k: v for k, v in schedule.items() if v is not None}

        network.set_optimizer(optimizer)
        network.set_lr_schedule(schedule)

    # create an example input for verification and summarization of model
    example_input = network.generate_example_input(
        include_batch_dimension=True, batch_size=8
    )

    # verify model
    if cfg.verify_model:
        verification = BatchGradientVerification(network)
        if not verification.check(input_array=example_input):
            raise ValueError(
                "Your model is mixing data across the batch dimension."
                " This can lead to wrong gradient updates in the optimizer."
                " Check the operations that reshape and permute tensor dimensions in your model."
            )

        # summarize model
        if isinstance(example_input, tuple):
            summary(
                network,
                *example_input,
                print_summary=True,
                max_depth=None,
            )
        else:
            summary(
                network,
                example_input,
                print_summary=True,
                max_depth=None,
            )

    return network


def construct_logger(cfg: DictConfig):
    if cfg.use_cometml:
        logger = CometLogger(
            project_name=cfg.project_name,
            experiment_name=cfg.experiment_name,
        )
    else:
        logger = True

    return logger


def construct_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks = []

    callback_cfg: DictConfig = cfg.callbacks

    ModelCheckpoint.CHECKPOINT_NAME_LAST = callback_cfg.get(
        "last_checkpoint_pattern", "last"
    )

    for cb_key in callback_cfg.to_add:
        if cb_key is None:
            continue

        if cb_key in callback_cfg:
            cb = hydra.utils.instantiate(callback_cfg[cb_key])
            log.info(f"Using callback <{cb}>")

            callbacks.append(hydra.utils.instantiate(callback_cfg[cb_key]))

    return callbacks


def construct_profiler(cfg: DictConfig):
    profile_cfg = cfg.get("profiler", None)

    if profile_cfg is None:
        return None
    else:
        return instantiate(profile_cfg)


################################################################################
# entrypoint of hydra script which does training and evaluation


def run_train_eval_script(cfg: DictConfig):
    # print config
    print(OmegaConf.to_yaml(cfg))
    print(f"{comet_ml.__version__}")
    print(f"{pytorch_lightning.__version__=}")
    print(f"{torch.__version__=}")

    # set random seed for python random module, numpy and pytorch
    pl.seed_everything(cfg.seed, workers=True)

    # create data module
    dm = construct_data_module(cfg)

    # create evaluator (for speaker recognition)
    evaluator: SpeakerRecognitionEvaluator = instantiate(cfg.evaluator)

    # create network module
    module = construct_module(cfg, evaluator, dm)

    # create logger
    logger = construct_logger(cfg)

    # create callbacks
    callbacks = construct_callbacks(cfg)

    # construct profiler
    profiler = construct_profiler(cfg)

    # create training/evaluator
    trainer: pl.Trainer = instantiate(
        cfg.trainer, logger=logger, callbacks=callbacks, profiler=profiler
    )

    # add tag if using cometml
    if hasattr(trainer.logger, "experiment") and hasattr(
        trainer.logger.experiment, "add_tag"
    ):
        trainer.logger.experiment.add_tag(cfg.tag)

    # tune model
    if cfg.tune_model:
        import matplotlib.pyplot as plt

        print(f"tuning for {cfg.tune_iterations} iterations")
        results = trainer.tune(
            module,
            datamodule=dm,
            lr_find_kwargs={
                "num_training": cfg.tune_iterations,
                "mode": "exponential",
                "early_stop_threshold": 3,
                "update_attr": True,
            },
        )

        if "lr_find" in results:
            result = results["lr_find"]
            filename = f"lr_find_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

            # save data points
            with open(f"{filename}.json", "w") as f:
                json.dump(
                    {
                        "lr_min": result.lr_min,
                        "lr_max": result.lr_max,
                        "num_training": result.num_training,
                        "data": {**result.results},
                    },
                    f,
                )

            # save figure
            fig = result.plot(suggest=True, show=True)
            fig.savefig(f"{filename}.png")

        return

    # train model
    if cfg.fit_model:
        trainer.fit(module, datamodule=dm)

    # test model
    if cfg.trainer.accelerator == "ddp":
        torch.distributed.destroy_process_group()

        if not trainer.global_rank == 0:
            return

    # create a new trainer which uses at most 1 gpu
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        gpus=min(1, int(cfg.trainer.get("gpus"))),
        accelerator=None,
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
    )

    result = None
    if cfg.eval_model and cfg.fit_model:
        # this will select the checkpoint with the best validation metric
        # according to the ModelCheckpoint callback
        try:
            result = trainer.test(datamodule=dm)
        except:
            # there might not have been a validation epoch
            result = trainer.test(module, datamodule=dm)
    elif cfg.eval_model:
        # this will simply test the given spk_module weights (when it's e.g
        # manually loaded from a checkpoint)
        result = trainer.test(module, datamodule=dm)

    if result is not None:
        if isinstance(result, list):
            if len(result) != 1:
                raise ValueError("expected result list to have length 1")

            result_obj = result[0]

            if "eer" in result_obj:
                objective = result_obj["eer"]
            elif "wer" in result_obj:
                objective = result_obj["wer"]
            else:
                raise ValueError(
                    f"unknown objective value out of keys "
                    f"{[k for k in result_obj.keys()]}"
                )

            return objective
        else:
            raise ValueError(f"result object has unknown type {type(result)=}")

    return None
