import sys
sys.path.append('src')

import argparse
import datetime

import torch
import tqdm

from torch.utils.data import (
    DataLoader,
)
from torch.utils.tensorboard import SummaryWriter

from training.input import parse_yml_config
from training.preparations import build_dataset, build_model


def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '-cf', '--config_file',
        help="Path to the yaml file containing the training configuration",
        type=str,
        default='train_config.yml'
    )
    return parser.parse_args()


parser = argparse.ArgumentParser(
    description="""Training script for Music Genre Classficiation
The script will parse all training arguments from your configuration file
and run training accordingly."""
)
args = parse_args(parser)
configs = parse_yml_config(args.config_file)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# build dataset
train_set, test_set = build_dataset(configs['data_args'])
train_loader = DataLoader(train_set, configs['training_args']['batch_size'], drop_last=True)
test_loader = DataLoader(test_set, configs['training_args']['batch_size'], drop_last=True)

# build model
model, optimizer = build_model(
    len(train_set.dataset._genre_to_id),
    configs['model']['backbone'],
    configs['training_args']['learning_rate'],
    configs['training_args']['optimizer'],
    device='cuda',
    inner_channels=configs['model']['inner_channels'],
    downsampling_rates=configs['model']['downsampling_rates'],
    num_linear_layers=configs['model']['num_linear_layers'],
)

gradient_scaler = torch.amp.grad_scaler.GradScaler(enabled=configs['training_args']['mixed_precision'])

if configs['training_args']['distributed_training']:
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
if configs['inout']['checkpoint']:
    ckpt = torch.load(configs['inout']['checkpoint'], weights_only=True)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    gradient_scaler.load_state_dict(ckpt['gradient_scaler'])

cpe_loss = torch.nn.CrossEntropyLoss()
loss_fn = lambda y_hat, y: (
    cpe_loss(y_hat, y) +
    configs['training_args']['regularization_lambda']
        * sum(w.pow(2).sum() for w in model.parameters())
)


# run training
tb_logger = SummaryWriter(
    f"{configs['inout']['ckpt_dir']}/{configs['inout']['logdir']}"
)
pbar = tqdm.tqdm(
    (
        range(1 + ckpt['epoch'], configs['training_args']['epochs'] + ckpt['epoch'] + 1)
        if configs['inout']['checkpoint'] else
        range(1, configs['training_args']['epochs'] + 1)
    ),
    desc='Epoch'
)
for epoch in pbar:
    train_loss, train_acc, test_loss, test_acc = 0, 0, 0, 0

    model.train()
    for step, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        with torch.autocast(
            device_type='cuda', dtype=torch.float16,
            enabled=configs['training_args']['mixed_precision']
        ):
            preds = model(data)
            loss = loss_fn(preds, labels)

        gradient_scaler.scale(loss).backward()
        gradient_scaler.step(optimizer)
        gradient_scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        train_acc += (labels == preds.argmax(dim=1)).sum().item()
        pbar.set_postfix({'loss': loss.item()})
        tb_logger.add_scalar(
            'step/train_loss',
            loss, (epoch - 1) * len(train_loader) + step
        )
        tb_logger.add_scalar(
            'step/learning_rate',
            configs['training_args']['learning_rate'],
            (epoch - 1) * len(train_loader) + step
        )

    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            with torch.autocast(
                device_type='cuda', dtype=torch.float16,
                enabled=configs['training_args']['mixed_precision']
            ):
                preds = model(data)
                test_loss += loss_fn(preds, labels).item()

            test_acc += (labels == preds.argmax(dim=1)).sum().item()

    tb_logger.add_scalars(
        'epoch/loss',
        {
            'train': train_loss / len(train_loader),
            'test': test_loss / len(test_loader),
        },
        epoch
    )
    tb_logger.add_scalars(
        'epoch/accuracy',
        {
            'train': train_acc / len(train_set),
            'test': test_acc / len(test_set),
        },
        epoch
    )
    tb_logger.flush()

tb_logger.close()


now = datetime.datetime.now()
timestamp = (
    f"{(now.year % 100):0>2}{now.month:0>2}{now.day:0>2}-"
    f"{now.hour:0>2}{now.minute:0>2}{now.second:0>2}"
)

torch.save({
    'epoch': epoch,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'gradient_scaler': gradient_scaler.state_dict()
}, f"{configs['inout']['ckpt_dir']}/{timestamp}_cnn.pth")
