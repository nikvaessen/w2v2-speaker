################################################################################
#
# Apply dropout on time and channel dimensions of wav2vec2 embedding
# as described in https://arxiv.org/abs/2006.11477
#
# Author(s): Nik Vaessen
################################################################################

from typing import List

import torch as t
import torch.nn as nn

################################################################################
# implementation as nn module


class EmbeddingMasker(nn.Module):
    def __init__(
        self,
        timestep_mask_prob: float,
        timestep_mask_width: int,
        channel_mask_prob: float,
        channel_mask_width: int,
        time_dim: int = 1,
        embedding_dim: int = 2,
    ):
        if not (0 <= channel_mask_prob <= 1):
            raise ValueError(
                f"probability channel_mask_prob {channel_mask_prob} expected to "
                f"be in range [0,1]"
            )
        if not (0 <= timestep_mask_prob <= 1):
            raise ValueError(
                f"probability timestep_mask_prob {timestep_mask_prob} expected to "
                f"be in range [0,1]"
            )

        if time_dim == 0 or embedding_dim == 0:
            raise ValueError("dimensions to mask cannot be dim 0 (batch dimension)")

        super().__init__()

        self.timestep_mask_prob = timestep_mask_prob
        self.timestep_mask_width = timestep_mask_width
        self.channel_mask_prob = channel_mask_prob
        self.channel_mask_width = channel_mask_width

        self.time_dim = time_dim
        self.embedding_dim = embedding_dim

    def forward(self, embedding_tensor: t.Tensor):
        if not self.training or (self.timestep_mask_prob + self.channel_mask_prob == 0):
            return embedding_tensor

        assert len(embedding_tensor.shape) == 3

        num_time_steps = embedding_tensor.shape[self.time_dim]
        num_channels = embedding_tensor.shape[self.embedding_dim]

        # create mask with same shape of embedding tensor
        m = t.ones(embedding_tensor.shape, device=embedding_tensor.device)

        # determine which time steps to mask
        if self.timestep_mask_prob > 0:
            time_masked = t.rand((num_time_steps,))
            time_masked = (
                t.where(
                    time_masked <= self.timestep_mask_prob, t.Tensor([0]), t.Tensor([1])
                )
                .numpy()
                .tolist()
            )

            time_masked = self.expand_mask(time_masked, self.timestep_mask_width)
            self.insert_into_mask(m, time_masked, 0, self.time_dim)

        # determine which channels to mask
        if self.timestep_mask_prob > 0:
            channel_mask = t.rand((num_channels,))
            channel_mask = (
                t.where(
                    channel_mask <= self.channel_mask_prob, t.Tensor([0]), t.Tensor([1])
                )
                .numpy()
                .tolist()
            )

            channel_mask = self.expand_mask(channel_mask, self.channel_mask_width)
            self.insert_into_mask(m, channel_mask, 0, self.embedding_dim)

        # mask and return the embedding
        return m * embedding_tensor

    @staticmethod
    def insert_into_mask(
        mask_tensor: t.Tensor, mask_list: List[int], mask_value: int, dim: int
    ):
        mask_idx = [idx for idx, value in enumerate(mask_list) if value == mask_value]

        if dim == 1:
            mask_tensor[:, mask_idx, :] = mask_value
        else:
            mask_tensor[:, :, mask_idx] = mask_value

        return mask_tensor

    @staticmethod
    def expand_mask(
        mask_list: List[int], mask_width: int, mask_value_to_expand: int = 0
    ):
        # repeat mask widths
        mask_idx = []

        for idx, mask_value in enumerate(mask_list):
            if mask_value == mask_value_to_expand:
                mask_idx.append(idx)

        expanded_mask_list = t.Tensor(mask_list)
        for idx in mask_idx:
            expanded_mask_list[idx : (idx + mask_width)] = mask_value_to_expand

        return expanded_mask_list.numpy().tolist()
