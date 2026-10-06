# Trabajo Práctico — Semana 6
## Arquitecturas CNN: Comparación, Batch Normalization y Transfer Learning

Curso: Introducción a Deep Learning. Clasificación de células sanguíneas (4 clases) con LeNet-5, VGG-11 (con y sin Batch Normalization) y transfer learning con ResNet-18 (PyTorch).

## Dataset
[Blood Cell Images (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/blood-cells), carpeta `dataset2-master`: EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL. TRAIN 9957 / TEST 2487 imágenes. Validación: 15 % de TRAIN (estratificado, semilla 42).

## Requisitos
Python 3.12, PyTorch 2.6 (CUDA 12.4 recomendado). Probado en una RTX 4050 Laptop (6 GB).
```
pip install -r requirements.txt
```

## Instalación y ejecución
1. Descargar los datos: `bash data/download_data.sh` (deja el dataset en `data/kagglehub/`).
2. Ejecutar los notebooks en orden desde `notebooks/` (ya incluyen sus resultados ejecutados):

| Notebook | Contenido | Tiempo aprox. (GPU) |
|---|---|---|
| `01_eda.ipynb` | EDA, estadísticas de normalización, similitud entre splits | 1 min |
| `02_lenet_vgg.ipynb` | Tarea 1: LeNet-5 y VGG-11, con/sin BN, 3 semillas | 10 min |
| `02b_bn_analysis.ipynb` | Tarea 2: BN en VGG-11 con 3 learning rates | 20 min |
| `03_transfer.ipynb` | Tarea 3: ResNet-18 con 3 estrategias | 40 min |

Reproducibilidad: semillas 42/43/44, cuDNN determinista. Misma semilla → mismas curvas.
Nota (Windows): en los notebooks `pandas` se importa antes que `torch`; en el orden inverso el kernel de Jupyter se cae al usar `DataFrame.to_csv`.

## Estructura
```
data/         script de descarga (el dataset no se versiona)
notebooks/    01_eda, 02_lenet_vgg, 02b_bn_analysis, 03_transfer
src/          models.py, train.py, utils.py
results/      tablas (csv/json) y figures/
informe.md    borrador de datos del informe (informe.pdf pendiente)
```

## Resultados (media ± desviación, 3 semillas; test con el mejor checkpoint de validación)

| Modelo | Parámetros | Test acc |
|---|---|---|
| LeNet-5 | 337 976 | 0.657 ± 0.066 |
| LeNet-5 + BN | 337 998 | 0.622 ± 0.079 |
| VGG-11 | 3 095 748 | 0.473 ± 0.289 (no entrena en 2 de 3 semillas) |
| VGG-11 + BN | 3 097 124 | 0.851 ± 0.016 |
| ResNet-18, feature extraction | 11.2 M (2 052 entrenables) | 0.422 ± 0.014 |
| ResNet-18, fine-tuning parcial | 11.2 M (10.5 M entrenables) | 0.825 ± 0.008 |
| ResNet-18, fine-tuning total | 11.2 M | **0.860 ± 0.004** |

## Principales hallazgos
- **BN estabiliza el entrenamiento de VGG-11:** con lr 1e-3, sin BN solo 1 de 3 semillas entrena; con BN las 3. Con lr 1e-4, BN llega al 80 % de validación en 6.0 épocas frente a 8.3.
- **BN no permitió un LR mayor:** con lr 1e-2 (Adam) ninguna variante entrena.
- **En LeNet, BN no mejora el test:** la diferencia está dentro de la variación entre semillas.
- **Fine-tuning total** es la mejor estrategia (0.860) y la más estable, pero supera por poco a VGG-11+BN desde cero (0.851) y cuesta ~10× más por época.
- **Brecha validación → test:** validación 95–100 % frente a test 47–86 %. Val se parece algo más a TRAIN que test (similitud coseno mediana 0.952 vs 0.945, `01_eda.ipynb`); sin duplicados exactos. Evidencia moderada, causa no demostrada.

## Referencias
LeCun et al. (1998); Krizhevsky et al. (2012); Simonyan & Zisserman (2014); He et al. (2016); Ioffe & Szegedy (2015).
