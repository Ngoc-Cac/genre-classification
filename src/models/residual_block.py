from torch import nn

from .structs import _ACT_FN


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
            _ACT_FN.get(activation_fn, nn.ReLU)()
            if isinstance(activation_fn, str) else
            activation_fn
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride, 1),
            nn.BatchNorm1d(in_channels),
            self.activation_fn,
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm1d(out_channels)
        )

        self.residual_connection = (
            (lambda x: x) if stride == 1 else
            nn.Conv1d(
                in_channels, out_channels,
                1, stride
            )
        )

    def forward(self, waveforms):
        conv_res = self.conv_block(waveforms)
        skip_res = self.residual_connection(waveforms)
        return self.activation_fn(conv_res + skip_res)


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
        inner_channels = in_channels * channel_mult // 4

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, inner_channels, 1),
            nn.BatchNorm1d(inner_channels), activation_fn,
            nn.Conv1d(inner_channels, inner_channels, kernel_size, stride, 1),
            nn.BatchNorm1d(inner_channels), activation_fn,
            nn.Conv1d(inner_channels, out_channels, 1)
        )
