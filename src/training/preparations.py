import os, warnings
import librosa
import torch

from torch.utils.data import Subset
from torchaudio.transforms import AmplitudeToDB

from models import GenreClassifier
from .structs import (
    DATASETS,
    FEATURE_TYPES,
    OPTIMIZERS,
    OPTIMIZERS_8BIT,
    WINDOW_FUNCTIONS,
)

from typing import Literal


def build_dataset(
    data_args: dict,
    feat_args: dict
) -> tuple[Subset, Subset]:
    root = data_args['root'][-1]
    root = f"{root}/{os.listdir(root)[0]}"
    _, sr = librosa.load(f"{root}/{os.listdir(root)[0]}", sr=None, duration=1)

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
    amp_to_db = AmplitudeToDB(top_db=80)
    def build_feat(wave, _):
        # don't convert to log scale if already mfcc
        spec = spec_builder(wave).unflatten(0, (1, -1))
        if feat_type != 'mfcc':
            spec = amp_to_db(spec)
        return (spec - spec.min()) / (spec.max() - spec.min())

    kwargs = {
        "sampling_rate": data_args['sampling_rate'],
        "preprocessor": build_feat
    }
    if data_args['type'] == 'fma':
        kwargs['subset_ratio'] = data_args['subset_ratio']

    dataset = DATASETS[data_args['type']](
        *data_args['root'], data_args['first_n_secs'],
        data_args['random_crops'], **kwargs
    )
    return dataset.random_split([data_args['train_ratio'], 1 - data_args['train_ratio']])


def build_model(
    num_labels: int,
    model_config_file: str,
    optimizer_args: dict,
    *,
    device: Literal['cuda', 'cpu'] = 'cpu',
    distirbuted_training: bool = False
) -> tuple[GenreClassifier, torch.optim.Optimizer]:
    model = GenreClassifier(1, num_labels, model_config_file).to(device)
    optimizer = (
        OPTIMIZERS_8BIT if optimizer_args['use_8bit_optimizer'] else OPTIMIZERS
    )[optimizer_args['type']]

    if distirbuted_training:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    else:
        warnings.warn(
            "distributed_training=true but only one GPU found! "
            "Running on single GPU instead...",
            RuntimeWarning
        )

    return model, optimizer(model.parameters(), **optimizer_args['kwargs'])
