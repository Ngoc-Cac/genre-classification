import torch

from torch import nn

from .utils import parse_model


class GenreClassifier(nn.Module):
    def __init__(self,
        input_channel: int,
        num_genres: int,
        model_conf: str,
        *,
        freq_as_channel: bool = False
    ):
        super().__init__()
        backbone, head = parse_model(
            model_conf, input_channel, num_genres, freq_as_channel
        )

        adapt_pool = nn.AdaptiveAvgPool1d if freq_as_channel else nn.AdaptiveAvgPool2d
        self.networks = nn.ModuleDict({
            "backbone": backbone,
            "global_avg_pooling": nn.Sequential(adapt_pool(1), nn.Flatten()),
            "classification_head": head
        })

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
