import os, wave

from torch import optim
from torch.utils.data import Subset
from torchaudio.transforms import AmplitudeToDB

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
    amp_to_db = AmplitudeToDB(top_db=80)
    def build_feat(wave, _):
        # don't convert to log scale if already mfcc
        spec = spec_builder(wave).unflatten(0, (1, -1))
        if feat_type != 'mfcc':
            spec = amp_to_db(spec)
        return (spec - spec.min()) / (spec.max() - spec.min())

    dataset = GTZAN(
        data_args['root'], data_args['first_n_secs'], data_args['random_crops'],
        preprocessor=build_feat
    )
    return dataset.random_split([data_args['train_ratio'], 1 - data_args['train_ratio']])


def build_model(
    num_labels: int,
    model_config_file: str,
    learning_rate: int | float,
    optimizer: _ALLOWED_OPTS,
    use_8bit_optimizers: bool = False,
    *,
    device: Literal['cuda', 'cpu'] = 'cpu'
) -> tuple[GenreClassifier, optim.Optimizer]:
    model = GenreClassifier(1, num_labels, model_config_file).to(device)
    optimizer = (
        OPTIMIZERS_8BIT if use_8bit_optimizers else OPTIMIZERS
    )[optimizer]
    return model, optimizer(model.parameters(), learning_rate)
