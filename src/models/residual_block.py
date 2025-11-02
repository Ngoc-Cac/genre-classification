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
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, 1),
            nn.BatchNorm2d(in_channels),
            self.activation_fn,
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.residual_connection = (
            (lambda x: x) if stride == 1 else
            nn.Conv2d(
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
        inner_channels = max(in_channels * channel_mult // 4, 1)  # in case results in 0

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1),
            nn.BatchNorm2d(inner_channels), self.activation_fn,
            nn.Conv2d(inner_channels, inner_channels, kernel_size, stride, 1),
            nn.BatchNorm2d(inner_channels), self.activation_fn,
            nn.Conv2d(inner_channels, out_channels, 1)
        )


class ResNet(nn.Module):
    def __init__(self,
        num_labels: int,
        in_channels: int,
        inner_channels: tuple[int],
        downsampling_rates: tuple[int],
        num_linear_layers: int = 0,
        *,
        kernel_size: int = 3,
        activation_fn: str | nn.Module = 'relu'
    ):
        super().__init__()

        activation_fn = (
            _ACT_FN.get(activation_fn, nn.ReLU)()
            if isinstance(activation_fn, str) else
            activation_fn
        )

        self.model = nn.Sequential()
        layer_iter = zip(inner_channels, downsampling_rates)
        for layer, (out_channels, downsampling_rate) in enumerate(layer_iter):
            self.model.add_module(
                f'bottleneck_{layer}',
                Bottleneck(
                    in_channels, out_channels // in_channels,
                    kernel_size, stride=downsampling_rate,
                    activation_fn=activation_fn
                )
            )
            in_channels = out_channels

        self.model.add_module(
            'global_pooling',
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        )

        for layer in range(num_linear_layers):
            self.model.add_module(
                f"linear_{layer}",
                nn.Sequential(
                    nn.Linear(out_channels, out_channels),
                    activation_fn
                )
            )

        self.model.add_module('output_logits', nn.Linear(out_channels, num_labels))

    def forward(self, waveforms):
        return self.model(waveforms)
