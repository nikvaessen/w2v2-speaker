################################################################################
#
# Implement temporal gating (squeeze-and-excitation layer) as described in:
#
# Wav2Spk: A Simple DNN Architecture for Learning Speaker Embeddings
# from Waveforms
#
# https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1287.pdf
#
# Author(s): Nik Vaessen
################################################################################

import torch as t
import torch.nn as nn

################################################################################
# pytorch module acting as temporal gate


class TemporalGate(nn.Module):
    def __init__(self, num_features: int):
        super(TemporalGate, self).__init__()

        self.W: nn.Parameter = nn.Parameter(
            nn.init.xavier_normal_(t.ones((num_features, num_features)))
        )
        self.b: nn.Parameter = nn.Parameter(
            nn.init.xavier_normal_(t.ones((num_features, 1)))
        )

    def forward(self, x):
        # we expect the input x to have dimensionality
        # [BS, NUM_FEATURES, NUM_FRAMES]
        # so that W matmul x results in the same shape
        mask = t.sigmoid(self.W.matmul(x) + self.b)

        return t.mul(mask, x)
