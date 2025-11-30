import os, itertools, wave
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

from typing import Literal, TypeAlias

_ALLOWED_OPTS: TypeAlias = Literal[*tuple(OPTIMIZERS.keys())]


def _normalize_spec(spec, is_mfcc=False):
    # don't convert to log scale if already mfcc
    if not is_mfcc:
        spec = torch.log(1 + spec)
    return (spec - spec.min()) / (spec.max() - spec.min())


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
    root = f"{data_args['root']}/{os.listdir(data_args['root'])[0]}"
    with wave.open(f"{root}/{os.listdir(root)[0]}") as wave_file:
        sr = wave_file.getframerate()

    feat_type = feat_args['feature_type']
    kwargs = {
        "n_fft": feat_args['n_fft'],
        "window_fn": WINDOW_FUNCTIONS[feat_args['window_type']],
    }
    if feat_type in ('mel', 'mfcc'):
        kwargs['n_mels'] = feat_args['n_mels']
    if feat_type == 'mfcc':
        kwargs = {"melkwargs": kwargs, "n_mfcc": feat_args['n_mfcc']}

    spec_builder = FEATURE_TYPES[feat_type](sr, **kwargs)
    dataset = GTZAN(
        data_args['root'], data_args['first_n_secs'], data_args['random_crops'],
        preprocessor=lambda wave, _: _normalize_spec(
            spec_builder(wave / abs(wave).max()).unflatten(0, (1, -1)),
            feat_type == 'mfcc'
        )
    )
    return _train_test_split(dataset, data_args['train_ratio'])


def build_model(
    num_labels: int,
    model_config_file: str,
    optimizer_args: dict,
    *,
    device: Literal['cuda', 'cpu'] = 'cpu'
) -> tuple[GenreClassifier, optim.Optimizer]:
    model = GenreClassifier(1, num_labels, model_config_file).to(device)
    optimizer = (
        OPTIMIZERS_8BIT if optimizer_args['use_8bit_optimizer'] else OPTIMIZERS
    )[optimizer_args['type']]
    return model, optimizer(model.parameters(), **optimizer_args['kwargs'])
