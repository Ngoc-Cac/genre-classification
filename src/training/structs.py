import torch

try:
    import bitsandbytes as bnb
except ModuleNotFoundError:
    bnb = None

from torchaudio.transforms import MelSpectrogram
from spectrograms import (
    Chromagram,
    LogFreqSpectrogram
)


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

OPTIMIZERS_8BIT = None if not bnb else {
    'adam': bnb.optim.Adam8bit,
    'adamw': bnb.optim.AdamW8bit,
    'sgd': bnb.optim.SGD8bit
}

WINDOW_FUNCTIONS = {
    'hann': torch.hann_window
}
