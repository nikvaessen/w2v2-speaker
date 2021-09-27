################################################################################
#
# This callback will monitor the RAM usage of each worker.
#
# Author(s): Nik Vaessen
################################################################################
from typing import Any

import psutil
import os

import pytorch_lightning as pl

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT


################################################################################
# callback implementation


class RamMemoryMonitor(pl.Callback):
    def __init__(self, frequency: int):
        self.frequency = frequency

        self.batches = 0

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.batches += 1

        if self.batches >= self.frequency:
            self.batches = 0

            try:
                self._monitor(trainer)
            except psutil.NoSuchProcess as e:
                pass

    @staticmethod
    def _monitor(trainer: pl.Trainer):
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)

        # track main process
        current_process_usage = _get_mem_usage_in_mb(current_process)

        # track child processes
        children_usage = [_get_mem_usage_in_mb(c) for c in children]

        # total usage
        total_usage = current_process_usage + sum(children_usage)

        # track usage
        if trainer is not None:
            trainer.logger.log_metrics(
                {
                    "mem_total": total_usage,
                }
            )


def _get_mem_usage_in_mb(p: psutil.Process):
    full_info = p.memory_full_info()

    # usage of process in bytes
    usage = full_info.uss

    # convert to megabytes
    usage = round(usage / float(1 << 20))

    return usage
