import os, warnings
import torch
import torchaudio

from torch.utils.data import Subset
from torchaudio.transforms import AmplitudeToDB

from models import GenreClassifier
from .structs import (
    DATASETS,
    FEATURE_TYPES,
    WINDOW_FUNCTIONS,
    OPTIMIZERS,
    OPTIMIZERS_8BIT,
    SCHEDULERS
)

from typing import Literal


def build_dataset(data_args: dict, feat_args: dict) -> tuple[Subset, Subset]:
    if data_args['sampling_rate']:
        sr = data_args['sampling_rate']
    else:
        root = data_args['root'][-1]
        root = f"{root}/{os.listdir(root)[0]}"
        _, sr = torchaudio.load(f"{root}/{os.listdir(root)[0]}")

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
        spec = spec_builder(wave)
        if feat_args['freq_as_channel']:
            spec = spec[0]
        if feat_type != 'mfcc':
            spec = amp_to_db(spec)
        return (spec - spec.mean()) / (spec.std() + 1e-6)
        # return (spec - spec.min()) / (spec.max() - spec.min())

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
    return dataset.random_split(
        [data_args['train_ratio'], 1 - data_args['train_ratio']]
        if data_args['train_ratio'] else None
    )


def build_model(
    num_labels: int,
    feature_args: dict,
    model_config_file: str,
    optimizer_args: dict,
    lr_scheduler_configs: dict,
    *,
    freq_as_channel: bool = False,
    device: Literal['cuda', 'cpu'] = 'cpu',
    distirbuted_training: bool = False
) -> tuple[GenreClassifier, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    in_channels = (
        1 if not freq_as_channel else
        12 if feature_args['feature_type'] == 'chroma' else
        128 if feature_args['feature_type'] == 'midi' else
        feature_args['n_mels'] if feature_args['feature_type'] == 'mel' else
        feature_args['n_mfcc'] 
    )

    model = GenreClassifier(
        in_channels, num_labels, model_config_file,
        freq_as_channel=freq_as_channel
    ).to(device)

    if distirbuted_training:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        else:
            warnings.warn(
                "distributed_training=true but only one GPU found! "
                "Running on single GPU instead...",
                RuntimeWarning
            )

    optimizer = (
        OPTIMIZERS_8BIT if optimizer_args['use_8bit_optimizer'] else OPTIMIZERS
    )[optimizer_args['type']](model.parameters(), **optimizer_args['kwargs'])

    warmup_start_factor = lr_scheduler_configs['warmup']['start_factor']
    warmup_steps = lr_scheduler_configs['warmup']['total_steps']
    decay_type = lr_scheduler_configs['decay']['type']

    warmup_scheduler = SCHEDULERS['linear'](
        optimizer, warmup_start_factor, total_iters=warmup_steps
    ) if warmup_steps else SCHEDULERS[None](optimizer)

    decay_scheduler = SCHEDULERS[decay_type](
        optimizer, **lr_scheduler_configs['decay']['kwargs']
    )

    return model, optimizer, torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, decay_scheduler], [warmup_steps]
    )
