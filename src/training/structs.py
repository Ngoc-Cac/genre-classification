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

WINDOW_FUNCTIONS = {
    'hann': torch.hann_window
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

SCHEDULERS = {
    'linear': torch.optim.lr_scheduler.LinearLR,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'exponential': torch.optim.lr_scheduler.ExponentialLR,
    'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    None: lambda opt, **_: torch.optim.lr_scheduler.ConstantLR(opt, 1, 0)
}
