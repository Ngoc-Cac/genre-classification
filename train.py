import sys
sys.path.append('src')

import argparse, datetime

import torch, tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training import (
    seed, parse_yml_config,
    build_dataset, build_model,
    train_loop, eval_loop
)
from training.logging import setup_logger


def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '-cf', '--config_file',
        help="Path to the yaml file containing the training configuration.",
        type=str,
        default='train_config.yml'
    )
    parser.add_argument(
        '-nw', '--num_workers',
        help="The number of workers to use for data loading tasks.",
        type=int,
        default=0
    )
    parser.add_argument(
        '-nv', '--no_verbose',
        help="Whether to disable the logging to STDOUT when running training.",
        action='store_true'
    )
    return parser.parse_args()


parser = argparse.ArgumentParser(
    description="""Training script for Music Genre Classification.
The script will parse all training arguments from your configuration file
and run training accordingly."""
)
now = datetime.datetime.now()
timestamp = (
    f"{(now.year % 100):0>2}{now.month:0>2}{now.day:0>2}-"
    f"{now.hour:0>2}{now.minute:0>2}{now.second:0>2}"
)
py_logger = setup_logger(__name__, f'logs/{timestamp}.log', to_stdout=True)

args = parse_args(parser)
py_logger.info(f"Parsing training configuration from {args.config_file}...")
configs = parse_yml_config(args.config_file)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed(configs['data_args']['seed'])

# build dataset
py_logger.info("Preparing the dataset...")
batch_size = configs['training_args']['batch_size']
train_set, test_set = build_dataset(configs['data_args'], configs['feature_args'])
train_loader, test_loader = DataLoader(
    train_set, batch_size, drop_last=True,
    num_workers=args.num_workers, persistent_workers=args.num_workers
), DataLoader(
    test_set, batch_size, drop_last=True,
    num_workers=args.num_workers, persistent_workers=args.num_workers
)

# build model
py_logger.info("Preparing the model...")
model, optimizer = build_model(
    len(train_set.dataset._genre_to_id),
    configs['inout']['model_path'],
    configs['training_args']['optimizer'],
    device=device,
    distirbuted_training=configs['training_args']['distributed_training']
)

gradient_scaler = torch.amp.grad_scaler.GradScaler(enabled=configs['training_args']['mixed_precision'])

if configs['inout']['checkpoint']:
    ckpt = torch.load(configs['inout']['checkpoint'], weights_only=True)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    gradient_scaler.load_state_dict(ckpt['gradient_scaler'])

loss_fn = torch.nn.CrossEntropyLoss()


# run training
mixed_prec = configs['training_args']['mixed_precision']
distributed = configs['training_args']['distributed_training']
optimizer_type = configs['training_args']['optimizer']['type']
total_epochs = configs['training_args']['epochs']
lr = configs['training_args']['optimizer']['kwargs']['lr']

if configs['training_args']['optimizer']['use_8bit_optimizer']:
    optimizer_type = "8-bit " + optimizer_type
py_logger.info(f"""
========== RUNNING TRAINING WITH CONFIGURATIONS ==========
Distributed training: {distributed}
Mixed-precision: {mixed_prec}
Total training | testing samples: {len(train_set)} | {len(test_set)}
Total batches per epoch: {len(train_loader)}
Epochs: {total_epochs}
Batch size: {batch_size}
Learning rate: {lr}
Optimizer: {optimizer_type}
=========================================================="""
)
tb_logger = SummaryWriter(
    f"{configs['inout']['ckpt_dir']}/{configs['inout']['logdir']}"
)
pbar = tqdm.tqdm(
    (
        range(1 + ckpt['epoch'], total_epochs + ckpt['epoch'] + 1)
        if configs['inout']['checkpoint'] else range(1, total_epochs + 1)
    ),
    desc='Epoch'
)
def log_train_step(step, loss):
    global pbar, tb_logger, epoch, train_loader, prev_loss, lr
    step = (epoch - 1) * len(train_loader) + step
    pbar.set_postfix({'loss': loss, 'test_loss': prev_loss})
    tb_logger.add_scalar('step/train_loss', loss, step)
    tb_logger.add_scalar('step/learning_rate', lr, step)

prev_loss = None
for epoch in pbar:
    model.train()
    train_loss, train_acc = train_loop(
        model, train_loader, loss_fn,
        optimizer, gradient_scaler, device,
        mixed_precision=mixed_prec,
        callback_fn=log_train_step
    )
    model.eval()
    test_loss, test_acc = eval_loop(
        model, test_loader, loss_fn, device=device,
        mixed_precision=mixed_prec
    )

    train_loss = train_loss / len(train_loader)
    prev_loss = test_loss = test_loss / len(test_loader)
    train_acc = train_acc / len(train_set)
    test_acc = test_acc / len(test_set)

    tb_logger.add_scalars(
        'epoch/loss', {'train': train_loss, 'test': test_loss}, epoch
    )
    tb_logger.add_scalars(
        'epoch/accuracy', {'train': train_acc, 'test': test_acc}, epoch
    )
    tb_logger.flush()

tb_logger.add_hparams(
    {
        'batch_size': batch_size, 'learning_rate': lr,
        'optimizer': optimizer_type,
        'feature_type': configs['feature_args']['feature_type']
    },
    {
        'hparam/train_accuracy': train_acc, 'hparam/test_accuracy': test_acc,
        'hparam/train_loss': train_loss, 'hparam/test_loss': test_loss
    }
)

tb_logger.close()
torch.save(
    {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'gradient_scaler': gradient_scaler.state_dict()
    },
    f"{configs['inout']['ckpt_dir']}/{timestamp}.pth"
)
