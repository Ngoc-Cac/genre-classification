import os
import wave
import torch

from torch import optim
from torch.utils.data import random_split, Subset

from data_utils.dataset import GTZAN
from models import GenreClassifier
from .structs import (
    FEATURE_TYPES,
    OPTIMIZERS,
    OPTIMIZERS_8BIT,
    WINDOW_FUNCTIONS,
)

from typing import Literal


def _normalize_spec(spec):
    db_spec = torch.log(1 + spec)
    return (db_spec - db_spec.min()) / (db_spec.max() - db_spec.min())


def build_dataset(
    data_args: dict
) -> tuple[Subset, Subset]:
    temp_file = f"{data_args['root']}/{os.listdir(data_args['root'])[0]}"
    temp_file = f"{temp_file}/{os.listdir(temp_file)[0]}"
    with wave.open(temp_file) as wave_file:
        sr = wave_file.getframerate()

    kwargs = {"window_fn": WINDOW_FUNCTIONS[data_args['window_type']]}
    if data_args['feature_type'] == 'mfcc':
        kwargs['n_mels'] = data_args['n_mels']

    spec_builder = FEATURE_TYPES[data_args['feature_type']](
        sr, data_args['n_fft'], **kwargs
    )
    dataset = GTZAN(
        data_args['root'],
        data_args['first_n_secs'],
        preprocessor=lambda wave, sr: _normalize_spec(
            spec_builder(wave / abs(wave).max()).unflatten(0, (1, -1))
        )
    )
    train_test_ratio = [data_args['train_ratio'], 1 - data_args['train_ratio']]
    return random_split(dataset, train_test_ratio)


def build_model(
    num_labels: int,
    backbone_type: Literal['cnn', 'resnet'],
    learning_rate: int | float,
    optimizer: Literal['adam', 'adamw', 'sgd'],
    use_8bit_optimizers: bool = False,
    *,
    device: Literal['cuda', 'cpu'] = 'cpu',
    **model_kwargs
) -> tuple[GenreClassifier, optim.Optimizer]:
    model = GenreClassifier(
        num_labels, 1, backbone_type=backbone_type,
        **model_kwargs
    ).to(device)

    optimizer = (
        OPTIMIZERS_8BIT if use_8bit_optimizers else OPTIMIZERS
    )[optimizer]

    return model, optimizer(model.parameters(), learning_rate)
