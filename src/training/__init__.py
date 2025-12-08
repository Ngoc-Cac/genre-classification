import numpy, random, torch

from training.input import parse_yml_config
from training.preparations import build_dataset, build_model
from training.loops import train_loop, eval_loop


def seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
