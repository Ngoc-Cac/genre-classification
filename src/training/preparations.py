import os
import itertools
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


def _train_test_split(
    dataset: GTZAN,
    train_ratio: float
) -> tuple[Subset, Subset]:
    # random_split doesnt actually check if dataset is a Dataset, it just neds a len method
    train_set, test_set = random_split(dataset._files, [train_ratio, 1 - train_ratio])
    train_idx, test_idx = train_set.indices, test_set.indices

    if (mult := dataset._rand_crops) > 1:
        train_idx = list(itertools.chain(*(
            [idx * mult + i for i in range(mult)] for idx in train_idx
        )))
        test_idx = list(itertools.chain(*(
            [idx * mult + i for i in range(mult)] for idx in test_idx
        )))
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def build_dataset(
    data_args: dict,
    feat_args: dict
) -> tuple[Subset, Subset]:
    temp_file = f"{data_args['root']}/{os.listdir(data_args['root'])[0]}"
    temp_file = f"{temp_file}/{os.listdir(temp_file)[0]}"
    with wave.open(temp_file) as wave_file:
        sr = wave_file.getframerate()

    kwargs = {"window_fn": WINDOW_FUNCTIONS[feat_args['window_type']]}
    if feat_args['feature_type'] == 'mfcc':
        kwargs['n_mels'] = feat_args['n_mels']

    spec_builder = FEATURE_TYPES[feat_args['feature_type']](
        sr, feat_args['n_fft'], **kwargs
    )
    dataset = GTZAN(
        data_args['root'], data_args['first_n_secs'], data_args['random_crops'],
        preprocessor=lambda wave, _: _normalize_spec(
            spec_builder(wave / abs(wave).max()).unflatten(0, (1, -1))
        )
    )
    return _train_test_split(dataset, data_args['train_ratio'])


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
