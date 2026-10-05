"""Bucle de entrenamiento y evaluación."""
import time

import torch
import torch.nn as nn


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += criterion(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
        if return_preds:
            preds.append(out.argmax(1).cpu())
            trues.append(y.cpu())
    if return_preds:
        return loss_sum / n, correct / n, torch.cat(trues), torch.cat(preds)
    return loss_sum / n, correct / n


def fit(model, train_loader, val_loader, epochs, lr, device, optimizer_fn=None,
        verbose=True):
    """Entrena y devuelve el historial por época.

    Claves: train_loss, train_acc, val_loss, val_acc, epoch_time.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_fn(params, lr) if optimizer_fn else torch.optim.Adam(params, lr=lr)
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_time")}
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        vl, va = evaluate(model, val_loader, criterion, device)
        for k, v in zip(hist, (tl, ta, vl, va, dt)):
            hist[k].append(v)
        if verbose:
            print(f"ep {ep:02d}/{epochs} | train {tl:.4f}/{ta:.4f} | "
                  f"val {vl:.4f}/{va:.4f} | {dt:.1f}s")
    return hist
