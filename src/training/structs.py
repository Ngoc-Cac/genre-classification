import torch

from torchaudio.transforms import MelSpectrogram
from spectrograms import (
    Chromagram,
    LogFreqSpectrogram
)

from models import CNNSpec, ResNet


FEATURE_TYPES = {
    'chroma': Chromagram,
    'midi': LogFreqSpectrogram,
    'mfcc': MelSpectrogram
}

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD
}

WINDOW_FUNCTIONS = {
    'hann': torch.hann_window
}

MODELS = {
    'cnn': CNNSpec,
    'resnet': ResNet
}