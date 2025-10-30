import os
import wave
import torch

from torch import optim
from torch.utils.data import random_split, Subset

from data_utils.dataset import GTZAN
from models import CNNSpec
from spectrograms import (
    Chromagram,
    LogFreqSpectrogram,
)

from typing import Literal


_FEAT_TYPES = {
    'chroma': Chromagram,
    'midi': LogFreqSpectrogram
}
_OPTIMIZERS = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sgd': optim.SGD
}

def build_dataset(
    data_args: dict
) -> tuple[Subset, Subset]:
    temp_file = f"{data_args['root']}/{os.listdir(data_args['root'])[0]}"
    temp_file = f"{temp_file}/{os.listdir(temp_file)[0]}"
    with wave.open(temp_file) as wave_file:
        sr = wave_file.getframerate()

    spec_builder = _FEAT_TYPES[data_args['feature_type']](
        sr, data_args['n_fft']
    )
    dataset = GTZAN(
        data_args['root'],
        data_args['first_n_secs'],
        preprocessor=lambda wave, sr: spec_builder(wave / abs(wave).max())
    )
    train_test_ratio = [data_args['train_ratio'], 1 - data_args['train_ratio']]
    return random_split(dataset, train_test_ratio)


def build_model(
    num_labels,
    feat_type,
    optimizer,
    learning_rate,
    act_fn='relu',
    *,
    device: Literal['cuda', 'cpu'] = 'cpu',
    checkpoint: str = ''
) -> tuple[CNNSpec, optim.Optimizer]:
    model = CNNSpec(
        num_labels,
        12 if feat_type == 'chroma' else 127,
        act_fn
    ).to(device)

    return model, _OPTIMIZERS[optimizer](model.parameters(), learning_rate)