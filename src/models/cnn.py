from torch import nn

from .structs import _ACT_FN


class CNNSpec(nn.Module):
    def __init__(self,
        num_labels: int,
        in_channels: int,
        inner_channels: int = (128, 128, 256, 256, 512),
        num_linear_layers: int = 3,
        kernel_size: int = 3,
        downsampling_rate: int = 2,
        activation_fn: str | nn.Module = 'relu'
    ):
        super().__init__()

        activation_fn = (
            _ACT_FN.get(activation_fn, nn.ReLU)()
            if isinstance(activation_fn, str) else
            activation_fn
        )

        self.model = nn.Sequential()
        for layer, out_channels in enumerate(inner_channels):
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size, padding=1
                ),
                nn.BatchNorm2d(out_channels), activation_fn,
                nn.AvgPool2d(
                    kernel_size, downsampling_rate, 1
                )
            )
            self.model.add_module(f'conv_block_{layer}', conv_block)
            in_channels = out_channels

        self.model.add_module(
            'global_avg_pool',
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
        )

        for layer in range(num_linear_layers):
            self.model.add_module(
                f'linear_{layer}',
                nn.Sequential(
                    nn.Linear(out_channels, out_channels),
                    activation_fn,
                )
            )

        self.model.add_module('output_logits', nn.Linear(out_channels, num_labels))

    def forward(self, spectrograms):
        return self.model(spectrograms)