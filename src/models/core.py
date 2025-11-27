import torch

from torch import nn

from .structs import ACT_FN
from .blocks import Conv2D, Bottleneck

from typing import Literal


class GenreClassifier(nn.Module):
    def __init__(self,
        num_labels: int,
        in_channels: int,
        inner_channels: tuple[int],
        downsampling_rates: tuple[int],
        num_linear_layers: int = 0,
        *,
        backbone_type: Literal['cnn', 'resnet'],
        kernel_size: int = 3,
        activation_fn: str | nn.Module = 'relu'
    ):
        super().__init__()

        activation_fn = (
            ACT_FN.get(activation_fn, nn.ReLU)()
            if isinstance(activation_fn, str) else
            activation_fn
        )

        self.networks = nn.ModuleDict({
            "backbone": self._build_backbone_net(
                in_channels, inner_channels, downsampling_rates,
                backbone_type, kernel_size, activation_fn
            ),
            "global_avg_pooling": nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()),
            "classification_head": self._build_classify_head(
                num_linear_layers, inner_channels[-1],
                inner_channels[-1], num_labels,
                activation_fn
            )
        })

    def _build_backbone_net(self,
        in_channels: int,
        inner_channels: tuple[int],
        downsampling_rates: tuple[int],
        backbone_type: Literal['cnn', 'resnet'],
        kernel_size: int,
        activation_fn: nn.Module,
    ):
        backbone = nn.Sequential()
        layer_iter = zip(inner_channels, downsampling_rates, strict=True)
        for layer, (out_channels, downsampling_rate) in enumerate(layer_iter):
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

    def _build_classify_head(self,
        num_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        activation_fn: nn.Module
    ):
        classify_head = nn.Sequential()
        for layer in range(num_layers):
            classify_head.add_module(
                f'fc_{layer}',
                nn.Sequential(nn.Linear(in_channels, hidden_channels), activation_fn)
            )
            in_channels = hidden_channels

        classify_head.add_module('output_logits', nn.Linear(hidden_channels, out_channels))
        return classify_head

    def forward(self, spectrograms: torch.Tensor):
        features = self.networks['backbone'](spectrograms)
        return self.networks['classification_head'](
            self.networks['global_avg_pooling'](features)
        )

    def __repr__(self):
        backbone = repr(self.networks['backbone']).replace('\n', '\n  ')
        global_avg_pooling = repr(self.networks['global_avg_pooling']).replace('\n', '\n  ')
        classification_head = repr(self.networks['classification_head']).replace('\n', '\n  ')
        return (
            "GenerClassifier(\n"
            f"  (backbone): {backbone}\n"
            f"  (global_avg_pooling): {global_avg_pooling}\n"
            f"  (classification_head): {classification_head}\n"
            ")"
        )
