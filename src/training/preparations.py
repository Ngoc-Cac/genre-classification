import os
import wave
import torch

from torch import optim
from torch.utils.data import random_split, Subset

from data_utils.dataset import GTZAN
from models import CNNSpec, ResNet
from .structs import (
    FEATURE_TYPES,
    MODELS,
    OPTIMIZERS,
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

    spec_builder = FEATURE_TYPES[data_args['feature_type']](
        sr, data_args['n_fft'],
        window_fn=WINDOW_FUNCTIONS[data_args['window_type']]
    )
    dataset = GTZAN(
        data_args['root'],
        data_args['first_n_secs'],
        preprocessor=lambda wave, sr: _normalize_spec(
            spec_builder(wave / abs(wave).max()).unflatten(0, (1, -1))
        )
    )
    train_test_ratio = [data_args['train_ratio'], 1 - data_args['train_ratio']]
    return random_split(
        dataset, train_test_ratio,
        torch.Generator().manual_seed(data_args['seed'])
    )


def build_model(
    num_labels: int,
    backbone_type: Literal['cnn', 'resnet'],
    learning_rate: int | float,
    optimizer: Literal['adam', 'adamw', 'sgd'],
    *,
    device: Literal['cuda', 'cpu'] = 'cpu',
    **model_kwargs
) -> tuple[CNNSpec | ResNet, optim.Optimizer]:
    model = MODELS[backbone_type](
        num_labels, 1, **model_kwargs
    ).to(device)
    return model, OPTIMIZERS[optimizer](model.parameters(), learning_rate)
