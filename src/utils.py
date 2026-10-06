"""Utilidades compartidas: semilla, rutas del dataset y conteo de parámetros."""
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # antes de iniciar CUDA

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (ROOT / "data" / "kagglehub" / "datasets" / "paultimothymooney"
            / "blood-cells" / "versions" / "6" / "dataset2-master"
            / "dataset2-master" / "images")
CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
FIG_DIR = ROOT / "results" / "figures"


MEAN_64 = (0.6791, 0.6418, 0.6610)
STD_64 = (0.2587, 0.2580, 0.2556)
MEAN_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET = (0.229, 0.224, 0.225)


class CellDataset(Dataset):
    """Imágenes precargadas en memoria (uint8). Flips aleatorios solo en train."""

    def __init__(self, images, labels, mean, std, augment=False):
        self.images, self.labels, self.augment = images, labels, augment
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i].float() / 255
        if self.augment:
            if random.random() < 0.5:
                x = x.flip(2)
            if random.random() < 0.5:
                x = x.flip(1)
        return (x - self.mean) / self.std, self.labels[i]


def _load_split(split, img_size):
    imgs, labels = [], []
    for k, c in enumerate(CLASSES):
        for f in sorted((DATA_DIR / split / c).glob("*.jpeg")):
            im = Image.open(f).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            imgs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
            labels.append(k)
    return torch.stack(imgs), torch.tensor(labels)


def get_loaders(img_size=64, batch_size=64, val_frac=0.15, mean=MEAN_64, std=STD_64,
                seed=SEED):
    """Devuelve train/val/test loaders. Val = val_frac de TRAIN, estratificado."""
    x, y = _load_split("TRAIN", img_size)
    xt, yt = _load_split("TEST", img_size)
    tr_idx, va_idx = train_test_split(np.arange(len(y)), test_size=val_frac,
                                      stratify=y.numpy(), random_state=seed)
    g = torch.Generator().manual_seed(seed)
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                        generator=g if shuffle else None)
    train = CellDataset(x[tr_idx], y[tr_idx], mean, std, augment=True)
    val = CellDataset(x[va_idx], y[va_idx], mean, std)
    test = CellDataset(xt, yt, mean, std)
    return mk(train, True), mk(val, False), mk(test, False)


def set_seed(seed=SEED, loader=None):
    """Fija todas las semillas y activa cuDNN determinista.

    Si se pasa el loader de train, también reinicia su generador de barajado
    (si no, el orden de los batches dependería de los entrenamientos previos).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if loader is not None:
        loader.generator.manual_seed(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
