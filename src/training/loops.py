import numpy as np
import torch

from sklearn.metrics import confusion_matrix

from typing import Any, Callable, Literal, TypeAlias

LOSS_FN_TYPE: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def train_loop(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: LOSS_FN_TYPE,
    optimizer: torch.optim.Optimizer,
    gradient_scaler: torch.GradScaler,
    device: Literal['cpu', 'cuda'],
    *,
    mixed_precision: bool = False,
    callback_fn: Callable[[int, float], Any] | None = None
) -> tuple[float, float]:
    if callback_fn is None:
        callback_fn = lambda *args: None
    train_loss, train_acc = 0, 0
    for step, (data, labels) in enumerate(dataloader):
        labels = labels.to(device)
        with torch.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=mixed_precision
        ):
            preds = model(data.to(device))
            loss = loss_fn(preds, labels)

        gradient_scaler.scale(loss).backward()
        gradient_scaler.step(optimizer)
        gradient_scaler.update()
        optimizer.zero_grad()

        loss = loss.item()
        train_loss += loss
        train_acc += (labels == preds.argmax(dim=1)).sum().item()
        callback_fn(step, loss)
    return train_loss, train_acc


@torch.no_grad()
def eval_loop(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: LOSS_FN_TYPE,
    device: Literal['cpu', 'cuda'],
    *,
    mixed_precision: bool = False,
    return_cm: bool = False
) -> tuple[float, float, np.ndarray | None]:
    test_loss = 0
    all_preds, all_labels = [], []

    for data, labels in dataloader:
        labels = labels.to(device)
        with torch.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=mixed_precision
        ):
            preds = model(data.to(device))
            loss = loss_fn(preds, labels).item()

        test_loss += loss
        preds = preds.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_labels, all_preds = torch.concat(all_labels), torch.concat(all_preds)
    return (
        test_loss, (all_labels == all_preds).sum().item(),
        confusion_matrix(all_labels, all_preds) if return_cm else None
    )