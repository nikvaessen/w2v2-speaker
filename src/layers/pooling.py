################################################################################
#
# Implement different pooling layers.
#
# They are used to reduce an embedding with
# a time-dimension (shape [BATCH_SIZE, NUM_FEATURES, NUM_FRAMES/NUM_SAMPLES],
# where num_frames or num_samples indicates some sequence in a time dimension
# which needs to be reduced to one.
#
# Author(s): Nik Vaessen
################################################################################

import random

import torch as t
import torch.nn as nn

from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling

################################################################################
# reduce by taking the mean across the time dimension


class MeanStatPool1D(nn.Module):
    def __init__(self, dim_to_reduce: int = 2):
        super().__init__()
        self.dim_to_reduce = dim_to_reduce

    def forward(self, tensor: t.Tensor):
        return t.mean(tensor, self.dim_to_reduce)


################################################################################
# reduce by taking the mean and standard deviation (which are stacked on
# top of each other)


class MeanStdStatPool1D(nn.Module):
    def __init__(self, dim_to_reduce: int = 2):
        super().__init__()
        self.dim_to_reduce = dim_to_reduce

    def forward(self, tensor: t.Tensor):
        return t.cat(t.std_mean(tensor, self.dim_to_reduce), 1)


################################################################################
# quantile pooling - use a stack with min, 0.25, 0.5, 0.75 and max quantile


class QuantilePool1D(nn.Module):
    def __init__(self, dim_to_reduce: int = 2):
        super().__init__()
        self.dim_to_reduce = dim_to_reduce
        self.quantiles = t.Tensor([0, 0.25, 0.5, 0.75, 1]).detach()

    def forward(self, tensor: t.Tensor):
        # input shape [BATCH, TIME, FEATURE]
        quantile_tensor = t.quantile(
            tensor, self.quantiles.to(tensor.device), dim=self.dim_to_reduce
        )  # quantile has shape [5, BATCH, FEATURE]

        # transform to expected shape [BATCH, FEATURE]
        quantile_tensor = t.transpose(quantile_tensor, 0, 1)
        quantile_tensor = t.flatten(quantile_tensor, start_dim=1, end_dim=2)

        return quantile_tensor


################################################################################
# max pooling


class MaxPool1D(nn.Module):
    def __init__(self, dim_to_reduce: int = 2):
        super().__init__()
        self.dim_to_reduce = dim_to_reduce

    def forward(self, tensor: t.Tensor):
        return t.max(tensor, dim=self.dim_to_reduce).values


################################################################################
# reduce by attending to the global context of the frames


class AttentiveStatPool1D(nn.Module):
    def __init__(self, embedding_size: int, dim_to_reduce: int = 2):
        super().__init__()
        self.pooling_layer = AttentiveStatisticsPooling(embedding_size)
        self.dim_to_reduce = dim_to_reduce

    def forward(self, tensor: t.Tensor):
        if self.dim_to_reduce == 2:
            pooled_embedding = self.pooling_layer(tensor)
        elif self.dim_to_reduce == 1:
            pooled_embedding = self.pooling_layer(tensor.transpose(1, 2))
        else:
            raise ValueError("can only pool dimension 1 or 2")

        pooled_embedding = pooled_embedding.squeeze()

        if len(pooled_embedding.shape) == 1:
            pooled_embedding = pooled_embedding[None, :]

        return pooled_embedding

################################################################################
# Index pooling - just take a particular index in the time dimension


class IndexPool1D(nn.Module):
    def __init__(self, selection_method: str, dim_to_reduce: int):
        super().__init__()
        self.selection_method = selection_method
        self.dim_to_reduce = dim_to_reduce

    def forward(self, tensor: t.Tensor):
        if self.selection_method == "first" or self.selection_method == "first+cls":
            view = self._select_first(tensor)
        elif self.selection_method == "middle":
            view = self._select_last(tensor)
        elif self.selection_method == "last":
            view = self._select_last(tensor)
        elif self.selection_method == "random":
            view = self._select_random(tensor)
        else:
            raise ValueError(f"unknown index {self.selection_method}")

        return t.clone(view)

    def _select_first(self, tensor: t.Tensor):
        if self.dim_to_reduce == 1:
            return tensor[:, 0, :]
        else:
            return tensor[:, :, 0]

    def _select_middle(self, tensor: t.Tensor):
        if self.dim_to_reduce == 1:
            return tensor[:, tensor.shape[1] // 2, :]
        else:
            return tensor[:, :, tensor.shape[2] // 2]

    def _select_last(self, tensor: t.Tensor):
        if self.dim_to_reduce == 1:
            return tensor[:, -1, :]
        else:
            return tensor[:, :, -1]

    def _select_random(self, tensor: t.Tensor):
        if self.dim_to_reduce == 1:
            return tensor[:, random.randint(0, int(tensor.shape[1]) - 1), :]
        else:
            return tensor[:, :, random.randint(0, int(tensor.shape[2]) - 1)]


################################################################################
# Placeholder for when no poolign is desired


class NoPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor: t.Tensor):
        return tensor
