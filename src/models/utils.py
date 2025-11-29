import yaml

from torch import nn

from .structs import ACT_FN as _VALID_ACT_FNS
from .blocks import Bottleneck, Conv2D, MLP

from typing import get_args, Iterable, Literal, TypeAlias


_BACKBONE_TYPES: TypeAlias = Literal['cnn', 'resnet']
_VALID_BACKBONES = get_args(_BACKBONE_TYPES)

_VALID_ACT_FNS = tuple(_VALID_ACT_FNS.keys())
_ACT_FN: TypeAlias = Literal[*_VALID_ACT_FNS]


def parse_model(
    config_file: str,
    input_channel: int,
    num_classes: int
) -> tuple[nn.Module, nn.Module]:
    with open(config_file, encoding='utf-8') as config_file:
        model_config = yaml.safe_load(config_file)

    _validate_model_config(model_config)

    backbone_conf, head_conf = model_config.values()
    kernel_sizes = backbone_conf['kernel_size']
    kernel_sizes = (
        kernel_sizes if isinstance(kernel_sizes, list) else
        [kernel_sizes] * len(backbone_conf['inner_channels'])
    )

    return (
        build_backbone_net(
            backbone_conf['type'], input_channel,
            backbone_conf['inner_channels'],
            backbone_conf['downsampling_rates'],
            kernel_sizes, backbone_conf['activation']
        ),
        MLP(
            backbone_conf['inner_channels'][-1],
            num_classes, head_conf['hidden_dims'],
            head_conf['dropout_probs'],
            activation_fn=head_conf['activation']
        )
    )


def build_backbone_net(
    backbone_type: _BACKBONE_TYPES,
    in_channels: int,
    inner_channels: Iterable[int],
    downsampling_rates: Iterable[int],
    kernel_sizes: Iterable[int],
    activation_fn: _ACT_FN,
) -> nn.Module:
    backbone = nn.Sequential()
    layer_iter = zip(
        inner_channels, downsampling_rates,
        kernel_sizes, strict=True
    )
    for layer, configs in enumerate(layer_iter):
        out_channels, downsampling_rate, kernel_size = configs

        if backbone_type == 'cnn':
            block_name = f'conv_{layer}'
            block = Conv2D(
                in_channels, out_channels,
                downsampling_rate, kernel_size,
                activation_fn=activation_fn
            )
        elif backbone_type == 'resnet':
            block_name = f'bottleneck_{layer}'
            block = Bottleneck(
                in_channels, out_channels // in_channels,
                kernel_size, stride=downsampling_rate,
                activation_fn=activation_fn
            )

        backbone.add_module(block_name, block)
        in_channels = out_channels
    return backbone


def _validate_positive_int(item, item_name: str):
    if not isinstance(item, int):
        type_ = type(item)
        raise TypeError(
            f"Found invalid type {type_} for the {item_name}! "
            f"{item_name.capitalize()} must be a positive interger."
        )
    elif item <= 0:
        raise ValueError(
            f"Found non-positive value for the {item_name}! "
            f"{item_name.capitalize()} must be a positive interger."
        )


def _validate_proba(item, item_name: str):
    if not isinstance(item, int | float):
        type_ = type(item)
        raise TypeError(
            f"Found invalid type {type_} for the {item_name} probability! "
            "Must be a number between 0 and 1."
        )
    elif not 0 <= item <= 1:
        raise ValueError(
            f"Found invalid value for the {item_name} probability! "
            "Must be a number between 0 and 1."
        )


def _validate_model_config(model_config: dict):
    if 'backbone' not in model_config:
        raise KeyError("Config file is missing the backbone configuration!")
    elif 'head' not in model_config:
        raise KeyError("Config file is missing the head configuration!")

    backbone = model_config['backbone']
    if backbone['type'] not in _VALID_BACKBONES:
        raise ValueError(
            f"Found invalid backbone type '{backbone['type']}'! "
            f"Backbone type must be one of {_VALID_BACKBONES}."
        )
    if backbone['activation'] not in _VALID_ACT_FNS:
        raise ValueError(
            f"Found invalid activation function '{backbone['activation']}'! "
            f"Activation function must be one of {_VALID_ACT_FNS}."
        )

    kernel_sizes = backbone['kernel_size']
    kernel_size_is_list = isinstance(kernel_sizes, list)
    if not kernel_size_is_list:
        kernel_sizes = [kernel_sizes] * len(backbone['inner_channels'])

    same_len = (
        len(backbone['inner_channels']) ==
        len(backbone['downsampling_rates']) ==
        len(kernel_sizes)
    )
    if not same_len:
        subject = (
            "inner_channels" +
            ("," if kernel_size_is_list else " and") +
            " downsampling_rates" +
            (" and kernel_size" if kernel_size_is_list else '')
        )
        lens = (len(backbone['inner_channels']), len(backbone['downsampling_rates']))
        if kernel_size_is_list:
            lens += (len(kernel_sizes),)
        raise ValueError(f"{subject} must have the same length! Found lengths: {lens}")

    for chan, down_rate, kernel_size in zip(
        backbone['inner_channels'],
        backbone['downsampling_rates'],
        kernel_sizes
    ):
        _validate_positive_int(chan, 'inner channel dimension')
        _validate_positive_int(down_rate, 'downsampling rate')
        _validate_positive_int(kernel_size, 'kernel size')

    head = model_config['head']
    if head['activation'] not in _VALID_ACT_FNS:
        raise ValueError(
            f"Found invalid activation function {head['activation']}! "
            f"Activation function must be one of {_VALID_ACT_FNS}."
        )

    same_len = len(head['hidden_dims']) == len(head['dropout_probs'])
    if not same_len:
        lens = [len(head['hidden_dims']), len(head['dropout_probs'])]
        if kernel_size_is_list:
            lens += len(kernel_sizes)
        raise ValueError(
            "hidden_dims and dropout_probs must have "
            f"the same length! Found lengths: {lens}"
        )

    for dim, prob in zip(head['hidden_dims'], head['dropout_probs']):
        _validate_positive_int(dim, 'hidden dimension of linear layers')
        _validate_proba(prob, "dropout")
