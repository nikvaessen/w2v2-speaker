################################################################################
#
# This callback implements visualization of the tensor(s) which are
# directly put into the model.
#
# It will:
#   1. print the raw input and ground truth tensor to a file
#   2. print some statistics of the input tensor to a file
#   3. try to convert the tensor back to a wav file and save it to disk
#
# Author(s): Nik Vaessen
################################################################################

import pathlib
import logging

from typing import Any, Optional, Dict

import torch
import torchaudio

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback

from src.data.modules.speaker.training_batch_speaker import (
    SpeakerClassificationDataBatch,
)
from src.util import debug_tensor_content

################################################################################
# callback implementation

# A logger for this file
log = logging.getLogger(__name__)


class InputMonitor(Callback):
    supported_batches = [SpeakerClassificationDataBatch]

    def __init__(self):

        self.logged_train_batch = False
        self.logged_val_batch = False
        self.logged_test_batch = False

    def on_train_batch_start(
        self,
        trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.logged_train_batch:
            return

        debug_log_batch(batch, name="train")
        self.logged_train_batch = True

    def on_validation_batch_start(
        self,
        trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.logged_val_batch:
            return

        debug_log_batch(batch, name="val")
        self.logged_val_batch = True

    def on_test_batch_start(
        self,
        trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.logged_test_batch:
            return

        debug_log_batch(batch, name="test")
        self.logged_test_batch = True


def debug_log_batch(
    batch: SpeakerClassificationDataBatch,
    save_folder: pathlib.Path = pathlib.Path.cwd() / "debug_batch",
    name: Optional[str] = None,
    additional_tensors: Optional[Dict[str, torch.Tensor]] = None,
    write_whole_tensor_to_file: bool = False,
):
    if type(batch) not in InputMonitor.supported_batches:
        raise ValueError(
            f"can only monitor one of {InputMonitor.supported_batches}, not {type(batch)}"
        )

    if name is None:
        log_dir = save_folder
    else:
        log_dir = save_folder / name

    log_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        f"dumping detailed logs of {name if name is not None else ''} batch to {log_dir}"
    )

    # actual values which go into the network and/or loss function
    debug_tensor_content(
        batch.network_input, "network_input", log_dir, write_whole_tensor_to_file
    )
    debug_tensor_content(
        batch.ground_truth, "ground_truth", log_dir, write_whole_tensor_to_file
    )

    # identifies of each sample in the batch
    with (log_dir / "keys.txt").open("w") as f:
        f.writelines([f"{k}\n" for k in batch.keys])

    # side information on transformation from wav to network input
    for key in batch.keys:
        side_info_dir = log_dir / key
        side_info = batch.side_info[key]

        if side_info is None:
            continue

        debug_tensor_content(
            side_info.original_tensor,
            "original_tensor",
            side_info_dir,
            write_whole_tensor_to_file,
        )
        torchaudio.save(
            side_info_dir / "original_tensor.wav", side_info.original_tensor, 16000
        )

        for idx, (transformed_tensor, debug_writer) in enumerate(
            side_info.pipeline_progress
        ):
            debug_writer.write(transformed_tensor, side_info_dir, idx)

    # write any additional tensors to the same folder
    if additional_tensors is None:
        return

    for name, tensor in additional_tensors.items():
        # avoid potential collision
        if name in ["keys", "ground_truth", "network_input"]:
            name += "_extra"

        debug_tensor_content(tensor, name, log_dir, write_whole_tensor_to_file)
