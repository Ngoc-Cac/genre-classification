from torch import nn

ACT_FN = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}

POOLING_TYPES = {
    'max': {
        1: nn.MaxPool1d,
        2: nn.MaxPool2d
    },
    'average': {
        1: nn.AvgPool1d,
        2: nn.AvgPool2d
    }
}
