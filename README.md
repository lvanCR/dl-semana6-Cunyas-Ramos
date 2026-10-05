# Trabajo Práctico — Semana 6
## Arquitecturas CNN: Comparación, Batch Normalization y Transfer Learning

Curso: Introducción a Deep Learning. Clasificación de células sanguíneas (4 clases) con LeNet-5, VGG-11 (con y sin BN) y Transfer Learning (ResNet-18 / VGG-16).

## Dataset
[Blood Cell Images (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/blood-cells): EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL.

## Requisitos
```
pip install -r requirements.txt
```

## Instalación y ejecución
1. Configurar credenciales de Kaggle (`~/.kaggle/kaggle.json`).
2. Descargar datos: `bash data/download_data.sh`
3. Ejecutar los notebooks en orden:
   - `notebooks/01_eda.ipynb`
   - `notebooks/02_lenet_vgg.ipynb`
   - `notebooks/03_transfer.ipynb`

Semilla fija: `42`. Compatible con Google Colab (GPU).

## Estructura
```
data/        descarga del dataset
notebooks/   EDA, LeNet/VGG, transfer learning
src/         models.py, train.py, utils.py
results/     métricas y figuras
informe.pdf  informe técnico (máx. 8 págs.)
```

## Resultados
| Modelo | Parámetros | Tiempo/época (s) | Acc. test |
|---|---|---|---|
| LeNet-5 | - | - | - |
| LeNet-5 + BN | - | - | - |
| VGG-11 | - | - | - |
| VGG-11 + BN | - | - | - |
| Feature Extraction | - | - | - |
| Fine-tuning parcial | - | - | - |
| Fine-tuning total | - | - | - |

## Principales hallazgos
_Pendiente._

## Referencias
LeCun et al. (1998); Krizhevsky et al. (2012); Simonyan & Zisserman (2014); He et al. (2016); Ioffe & Szegedy (2015).
