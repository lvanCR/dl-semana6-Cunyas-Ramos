"""Bucle de entrenamiento y evaluación."""
import time

import torch
import torch.nn as nn

from utils import count_params, set_seed


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


def run_experiment(model_fn, tag, train_loader, val_loader, test_loader, epochs, lr,
                   device, seed, ckpt_dir, optimizer_fn=None, verbose=False):
    """Entrena con una semilla, restaura el mejor checkpoint (val) y evalúa en test.

    model_fn: callable sin argumentos que devuelve un modelo nuevo.
    Devuelve dict con params, hist, test_loss, test_acc, y_true, y_pred.
    """
    set_seed(seed, train_loader)
    model = model_fn().to(device)
    ckpt = ckpt_dir / f"{tag}_s{seed}_best.pt"
    hist = fit(model, train_loader, val_loader, epochs, lr, device,
               optimizer_fn=optimizer_fn, verbose=verbose, ckpt_path=ckpt)
    model.load_state_dict(torch.load(ckpt))
    tl, ta, y, p = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, return_preds=True)
    return dict(tag=tag, seed=seed, params=count_params(model), hist=hist,
                test_loss=tl, test_acc=ta, y_true=y.tolist(), y_pred=p.tolist())


def fit(model, train_loader, val_loader, epochs, lr, device, optimizer_fn=None,
        verbose=True, ckpt_path=None):
    """Entrena y devuelve el historial por época.

    Si ckpt_path está definido, guarda allí el estado con mejor val_acc.

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
        if ckpt_path is not None and va >= max(hist["val_acc"]):
            torch.save(model.state_dict(), ckpt_path)
        if verbose:
            print(f"ep {ep:02d}/{epochs} | train {tl:.4f}/{ta:.4f} | "
                  f"val {vl:.4f}/{va:.4f} | {dt:.1f}s")
    return hist
