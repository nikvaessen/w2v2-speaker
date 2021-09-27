################################################################################
#
# Define a base lightning module for speech and/or speaker recognition network.
#
# Author(s): Nik Vaessen
################################################################################

import logging

from abc import abstractmethod
from typing import Callable, Optional

import torch as t
import torch.nn
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf


################################################################################
# Definition of speaker recognition API

# A logger for this file

log = logging.getLogger(__name__)


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        auto_lr_find: Optional[
            float
        ] = None,  # will be automatically passed by pytorch-lightning to children
    ):
        super().__init__()

        # input arguments
        self.loss_fn = loss_fn_constructor()

        # created by set_methods
        self.optimizer = None
        self.schedule = None
        self.warmup_optimizer = None
        self.warmup_schedule = None

        # flag determining which optimizer/schedule `configure_optimizers` uses
        self.warmup_enabled = False

        # auto_lr_find is set when you don't want to train the model
        # but want plot a learning rate against loss
        self.auto_lr_find = auto_lr_find

        # log hyperparameters
        self.save_hyperparameters(OmegaConf.to_container(hyperparameter_config))

    def set_optimizer(self, optimizer: t.optim.Optimizer):
        self.optimizer = optimizer

    def set_lr_schedule(self, schedule: t.optim.lr_scheduler._LRScheduler):
        self.schedule = schedule

    @abstractmethod
    def generate_example_input(
        self, include_batch_dimension: bool, batch_size: Optional[int]
    ):
        pass

    def configure_optimizers(self):
        if self.auto_lr_find:
            log.info("USING the `auto_lr_find` learning rate and optimizer!")
            return torch.optim.Adam(self.parameters(), lr=self.auto_lr_find)

        return [self.optimizer], [self.schedule]
