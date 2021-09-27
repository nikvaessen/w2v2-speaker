from .dummy import DummyModule, DummyModuleConfig
from .wav2spk import Wav2SpkModule, Wav2SpkModuleConfig
from .wav2vec2_fc import Wav2vec2FCModule, Wav2vec2FCModuleConfig
from .wav2vec2_paired_input import (
    Wav2vec2PairedSpeakerModule,
    Wav2vec2PairedSpeakerModuleConfig,
)
from .wav2vec_fc import Wav2vecFCModule, Wav2vecFCModuleConfig
from .wav2vec_xvector import Wav2vecXVectorModule, Wav2vecXVectorModuleConfig
from .xvector import XVectorModule, XVectorModuleConfig
from .ecapa_tdnn import EcapaTdnnModule, EcapaTDNNModuleConfig

from .paired_speaker_recognition_module import PairedSpeakerRecognitionLightningModule
from .speaker_recognition_module import SpeakerRecognitionLightningModule
