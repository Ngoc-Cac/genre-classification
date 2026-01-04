import sys
sys.path.append('src')

import argparse, datetime, textwrap

import matplotlib.pyplot as plt, torch, tqdm

from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay
)
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
        type=str, default='train_config.yml'
    )
    parser.add_argument(
        '-nw', '--num_workers',
        help="The number of workers to use for data loading tasks.",
        type=int, default=0
    )
    parser.add_argument(
        '-nv', '--no_verbose',
        help="Whether to disable the logging to STDOUT when running training.",
        action='store_true'
    )
    parser.add_argument(
        '-d', '--device',
        help="The available CUDA device.",
        type=str, default="cuda"
    )
    return parser.parse_args()


def draw_cm(
    labels: torch.Tensor,
    preds: torch.Tensor,
    cbar_offset: tuple[float, float] = (.02, .02)
) -> plt.Figure:
    global test_set
    fig, ax = plt.subplots(figsize=(6, 6))

    cmp = ConfusionMatrixDisplay.from_predictions(
        labels, preds, ax=ax,
        xticks_rotation=45, colorbar=False,
        display_labels=test_set.dataset.id_to_genre
    )
    cax = fig.add_axes([
        ax.get_position().x1 + cbar_offset[0], ax.get_position().y0,
        cbar_offset[1], ax.get_position().height
    ])
    plt.colorbar(cmp.im_,  cax=cax)
    return fig


def log_train_step(step, loss):
    global pbar, tb_logger, epoch, train_loader, postfix_dict
    step = (epoch - 1) * len(train_loader) + step
    postfix_dict['loss'] = loss
    pbar.set_postfix(postfix_dict)
    tb_logger.add_scalar('step/train_loss', loss, step)


now = datetime.datetime.now()
timestamp = (
    f"{(now.year % 100):0>2}{now.month:0>2}{now.day:0>2}-"
    f"{now.hour:0>2}{now.minute:0>2}{now.second:0>2}"
)
py_logger = setup_logger(__name__, f'logs/{timestamp}.log', to_stdout=True)

# set up except hook to handle any exceptions while running the event loop
def except_hook(exc_type, exc_value, exc_tb):
    py_logger.critical(
        'Program exited due to exception:',
        exc_info=(exc_type, exc_value, exc_tb)
    )
    sys.exit(1)
sys.excepthook = except_hook

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent("""\
              Music Genre Classification Training Script
        ------------------------------------------------------
        The script will parse all training arguments from your
           configuration file and run training accordingly.
    """)
)

args = parse_args(parser)
py_logger.info(f"Parsing training configuration from {args.config_file}...")
configs = parse_yml_config(args.config_file)
device = args.device if torch.cuda.is_available() else 'cpu'
seed(configs['data_args']['seed'])

# build dataset
py_logger.info("Preparing the dataset...")
batch_size = configs['training_args']['batch_size']
train_set, test_set = build_dataset(configs['data_args'], configs['feature_args'])
train_loader, test_loader = DataLoader(
    train_set, batch_size, shuffle=True,
    drop_last=(len(train_set) % batch_size == 1),
    num_workers=args.num_workers, persistent_workers=bool(args.num_workers)
), DataLoader(
    test_set, batch_size,
    num_workers=args.num_workers, persistent_workers=bool(args.num_workers)
)
id_to_genre = train_set.dataset.id_to_genre

# build model
py_logger.info("Preparing the model...")
model, optimizer, lr_scheduler = build_model(
    len(id_to_genre), configs['feature_args'],
    configs['inout']['model_path'],
    configs['optimizer'],
    configs['lr_schedulers'],
    freq_as_channel=configs['feature_args']['freq_as_channel'],
    device=device,
    distirbuted_training=configs['training_args']['distributed_training']
)

mixed_prec = configs['training_args']['mixed_precision']
gradient_scaler = torch.amp.grad_scaler.GradScaler(enabled=mixed_prec)

if configs['inout']['checkpoint']:
    ckpt = torch.load(configs['inout']['checkpoint'], weights_only=True)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    gradient_scaler.load_state_dict(ckpt['gradient_scaler'])
    lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

loss_fn = torch.nn.CrossEntropyLoss()


# run training
distributed = configs['training_args']['distributed_training']
total_epochs = configs['training_args']['epochs']
optimizer_type = configs['optimizer']['type']
lr = configs['optimizer']['kwargs']['lr']
postfix_dict = {'test_loss': None, 'test_acc': None, 'loss': None}

if configs['optimizer']['use_8bit_optimizer']:
    optimizer_type = "8-bit " + optimizer_type
py_logger.info(textwrap.dedent(f"""
    ========== RUNNING TRAINING WITH CONFIGURATIONS ==========
    Distributed training: {distributed} | Mixed-precision: {mixed_prec}
    Total training | testing samples: {len(train_set)} | {len(test_set)}
    Total batches per epoch: {len(train_loader)}
    Total Epochs: {total_epochs}
    Batch size: {batch_size} | Initial Learning rate: {lr}
    Optimizer: {optimizer_type}
    ==========================================================
"""))
tb_logger = SummaryWriter(
    f"{configs['inout']['ckpt_dir']}/{configs['inout']['logdir']}"
)
pbar = tqdm.tqdm(
    (
        range(1 + ckpt['epoch'], total_epochs + ckpt['epoch'] + 1)
        if configs['inout']['checkpoint'] else range(1, total_epochs + 1)
    ),
    postfix=postfix_dict, desc='Epoch'
)
for epoch in pbar:
    train_loss, train_acc = train_loop(
        model, train_loader, loss_fn,
        optimizer, gradient_scaler, device,
        mixed_precision=mixed_prec,
        callback_fn=log_train_step
    )
    test_loss, test_acc, tru_pred = eval_loop(
        model, test_loader, loss_fn, device=device,
        mixed_precision=mixed_prec, return_preds=True
    )
    lr_scheduler.step()

    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_set)
    postfix_dict['test_loss'] = test_loss = test_loss / len(test_loader)
    postfix_dict['test_acc'] = test_acc = test_acc / len(test_set)

    tb_logger.add_scalar('epoch/lr', lr_scheduler.get_last_lr()[0], epoch)
    tb_logger.add_scalars(
        'epoch/loss', {'train': train_loss, 'test': test_loss}, epoch
    )
    tb_logger.add_scalars(
        'epoch/accuracy', {'train': train_acc, 'test': test_acc}, epoch
    )
    tb_logger.add_figure(
        'epoch/confusion_mat', draw_cm(*tru_pred), epoch
    )
    tb_logger.add_text(
        'epoch/report',
        classification_report(
            *tru_pred, target_names=id_to_genre, zero_division=0.0
        )
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

if distributed:
    # unwrap the model if distributed before checkpointing
    model = model.module
torch.save(
    {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'gradient_scaler': gradient_scaler.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    },
    f"{configs['inout']['ckpt_dir']}/{timestamp}.pth"
)
