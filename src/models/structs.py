from torch import nn

ACT_FN = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}

POOLING_TYPES = {
    'max': nn.MaxPool2d,
    'average': nn.AvgPool2d
}