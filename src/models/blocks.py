import torch

from torch import nn

from .structs import ACT_FN

from typing import Iterable

__all__ = (
    'Conv2D',
    'Basic',
    'Bottleneck'
)


class MLP(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        dropout_probs: Iterable[int | float],
        *,
        activation_fn: str | nn.Module = 'relu',
    ):
        super().__init__()

        act_fn = (
            ACT_FN.get(activation_fn, nn.ReLU)()
            if isinstance(activation_fn, str) else activation_fn
        )
        self._repr = ["MLP("]
        self.mlp = nn.Sequential()

        prev_dim = input_dim
        layers = enumerate(zip(hidden_dims, dropout_probs, strict=True))
        for i, (dim, drop_prob) in layers:
            fc = nn.Sequential(nn.Linear(prev_dim, dim), act_fn)
            if 0 < drop_prob <= 1:
                fc.append(nn.Dropout(drop_prob))
            self.mlp.append(fc)
            self._repr.append(f"  (fc_{i}): " + repr(fc).replace('\n', '\n  '))
            prev_dim = dim

        output = nn.Linear(prev_dim, output_dim)
        self.mlp.append(output)

        self._repr.append("  (output_logits): " + repr(output).replace('\n', '\n  '))
        self._repr.append(")")
        self._repr = '\n'.join(self._repr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def __repr__(self):
        return self._repr


class Conv1D(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        activation_fn: str | nn.Module = 'relu',
        batch_norm: bool = True,
    ):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()
        self.act_fn = (
            ACT_FN.get(activation_fn, nn.ReLU)()
            if isinstance(activation_fn, str) else activation_fn
        )

        reprs = ["Conv2D(", "  (conv): " + repr(self.conv).replace('\n', '\n  ')]
        if batch_norm:
            reprs.append("  (batch_norm): " + repr(self.bn).replace('\n', '\n  '))
        reprs.append("  (activation): " + repr(self.act_fn).replace('\n', '\n  '))
        reprs.append(")")

        self._repr = '\n'.join(reprs)

    def forward(self, x: torch.Tensor):
        # conv -> bn -> act
        return self.act_fn(self.bn(self.conv(x)))

    def __repr__(self):
        return self._repr


class Conv2D(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        activation_fn: str | nn.Module = 'relu',
        batch_norm: bool = True,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.act_fn = (
            ACT_FN.get(activation_fn, nn.ReLU)()
            if isinstance(activation_fn, str) else activation_fn
        )

        reprs = ["Conv2D(", "  (conv): " + repr(self.conv).replace('\n', '\n  ')]
        if batch_norm:
            reprs.append("  (batch_norm): " + repr(self.bn).replace('\n', '\n  '))
        reprs.append("  (activation): " + repr(self.act_fn).replace('\n', '\n  '))
        reprs.append(")")

        self._repr = '\n'.join(reprs)

    def forward(self, x: torch.Tensor):
        # conv -> bn -> act
        return self.act_fn(self.bn(self.conv(x)))

    def __repr__(self):
        return self._repr


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
