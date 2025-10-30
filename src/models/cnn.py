from torch import nn

_ACT_FN = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}
class CNNSpec(nn.Module):
    def __init__(self,
        num_labels,
        in_channels: int = 12,
        act_fn: str | nn.Module = 'relu'
    ):
        super().__init__()

        if isinstance(act_fn, str):
            act_fn = _ACT_FN.get(act_fn, nn.ReLU)
        
        act_fn = act_fn()

        self.model = nn.Sequential(
            nn.Conv1d(12, 16, 3, 2), # 53
            act_fn,
            nn.Conv1d(16, 32, 3, 2), # 26
            act_fn,
            nn.Conv1d(32, 64, 3, 2), # 12
            act_fn,
            nn.Conv1d(64, 128, 3, 2), # 5
            act_fn,
            nn.Conv1d(128, 128, 3, 2), # 2
            act_fn,
            nn.Conv1d(128, 128, 1, 2), # 1
            act_fn,
            nn.Conv1d(128, 256, 1),
            act_fn,
            nn.Conv1d(256, num_labels, 1)
        )

    def forward(self, spectrograms):
        preds = self.model(spectrograms)

        return preds.flatten(start_dim=1) if self.training else preds