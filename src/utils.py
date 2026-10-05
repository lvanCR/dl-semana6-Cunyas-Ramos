"""Utilidades compartidas: semilla, rutas del dataset y conteo de parámetros."""
import random
from pathlib import Path

import numpy as np
import torch

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (ROOT / "data" / "kagglehub" / "datasets" / "paultimothymooney"
            / "blood-cells" / "versions" / "6" / "dataset2-master"
            / "dataset2-master" / "images")
CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
FIG_DIR = ROOT / "results" / "figures"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
