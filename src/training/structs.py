import torch

try:
    import bitsandbytes as bnb
except ModuleNotFoundError:
    bnb = None

from torchaudio.transforms import MelSpectrogram, MFCC

from datasets import FMA, GTZAN
from spectrograms import Chromagram, LogFreqSpectrogram


DATASETS = {
    "fma": FMA,
    "gtzan": GTZAN
}

FEATURE_TYPES = {
    'chroma': Chromagram,
    'midi': LogFreqSpectrogram,
    'mel': MelSpectrogram,
    'mfcc': MFCC
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
