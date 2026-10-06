#!/usr/bin/env bash
# Descarga el dataset Blood Cells (Kaggle) en data/kagglehub/, ruta que espera src/utils.py.
# Requiere: pip install kagglehub  (y credenciales de Kaggle si se solicitan).
set -e
cd "$(dirname "$0")"
export KAGGLEHUB_CACHE="$(pwd)/kagglehub"
python -c "import kagglehub; print(kagglehub.dataset_download('paultimothymooney/blood-cells'))"
