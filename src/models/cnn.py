from torch import nn

from .structs import _ACT_FN


class CNNSpec(nn.Module):
    def __init__(self,
        num_labels,
        in_channels: int = 12,
        activation_fn: str | nn.Module = 'relu'
    ):
        super().__init__()

        activation_fn = (
            _ACT_FN.get(activation_fn, nn.ReLU)()
            if isinstance(activation_fn, str) else
            activation_fn()
        )

        self.model = nn.Sequential(
            nn.Conv1d(12, 16, 3, 2), # 53
            activation_fn,
            nn.Conv1d(16, 32, 3, 2), # 26
            activation_fn,
            nn.Conv1d(32, 64, 3, 2), # 12
            activation_fn,
            nn.Conv1d(64, 128, 3, 2), # 5
            activation_fn,
            nn.Conv1d(128, 128, 3, 2), # 2
            activation_fn,
            nn.Conv1d(128, 128, 1, 2), # 1
            activation_fn,
            nn.Conv1d(128, 256, 1),
            activation_fn,
            nn.Conv1d(256, num_labels, 1)
        )

    def forward(self, spectrograms):
        preds = self.model(spectrograms)

        return preds.flatten(start_dim=1) if self.training else preds