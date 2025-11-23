import torch

from typing import Callable, Literal, Any


def train_loop(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
    for step, (data, labels) in enumerate(train_loader):
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

        train_loss += loss.item()
        train_acc += (labels == preds.argmax(dim=1)).sum().item()
        callback_fn(step, loss)
    return train_loss, train_acc


@torch.no_grad()
def eval_loop(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: Literal['cpu', 'cuda'],
    *,
    mixed_precision: bool = False
) -> tuple[float, float]:
    test_loss, test_acc = 0, 0
    for data, labels in test_loader:
        labels = labels.to(device)
        with torch.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=mixed_precision
        ):
            preds = model(data.to(device))
            test_loss += loss_fn(preds, labels).item()
        test_acc += (labels == preds.argmax(dim=1)).sum().item()
    return test_loss, test_acc