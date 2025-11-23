import torch

from torch import nn

from .structs import ACT_FN, POOLING_TYPES

from typing import Literal

__all__ = (
    'Conv2D',
    'Basic',
    'Bottleneck'
)


class Conv2D(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        downsampling_rate: int = 1,
        kernel_size: int = 3,
        *,
        activation_fn: str | nn.Module = 'relu',
        pooling_type: Literal['max', 'average'] = 'average'
    ):
        super().__init__()

        activation_fn = (
            ACT_FN.get(activation_fn, nn.ReLU)()
            if isinstance(activation_fn, str) else
            activation_fn
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            activation_fn
        )
        if downsampling_rate > 1:
            self.conv.append(
                POOLING_TYPES[pooling_type](
                    kernel_size, downsampling_rate, 1
                )
            )

    def forward(self, x: torch.Tensor):
        return self.conv(x)

    def __repr__(self):
        return repr(self.conv)


## ResNet blocks ##
class Basic(nn.Module):
    def __init__(self,
        in_channels: int,
        channel_mult: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        activation_fn: str | nn.Module = 'relu'
    ):
        super().__init__()

        out_channels = in_channels * channel_mult

        self.activation_fn = (
            ACT_FN.get(activation_fn, nn.ReLU)()
            if isinstance(activation_fn, str) else
            activation_fn
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, 1),
            nn.BatchNorm2d(in_channels),
            self.activation_fn,
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.residual_connection = (
            nn.Identity() if stride == channel_mult == 1 else
            nn.Conv2d(in_channels, out_channels, 1, stride)
        )

    def forward(self, waveforms):
        conv_res = self.conv_block(waveforms)
        skip_res = self.residual_connection(waveforms)
        return self.activation_fn(conv_res + skip_res)

    def __repr__(self):
        return super().__repr__()


class Bottleneck(Basic):
    def __init__(self,
        in_channels: int,
        channel_mult: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        activation_fn: str | nn.Module = 'relu'
    ):
        super().__init__(in_channels, channel_mult, kernel_size, stride, activation_fn)

        out_channels = in_channels * channel_mult
        inner_channels = max(in_channels * channel_mult // 4, 1)  # in case results in 0

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1),
            nn.BatchNorm2d(inner_channels), self.activation_fn,
            nn.Conv2d(inner_channels, inner_channels, kernel_size, stride, 1),
            nn.BatchNorm2d(inner_channels), self.activation_fn,
            nn.Conv2d(inner_channels, out_channels, 1)
        )
