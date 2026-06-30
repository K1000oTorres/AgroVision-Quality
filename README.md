# 🍎 AgroVision Quality

Sistema de detección automática del estado de maduración de frutas utilizando **YOLOv8**, **PyTorch** y **MLflow**.

El proyecto permite entrenar modelos de detección de objetos capaces de identificar diferentes tipos de frutas y su estado de maduración mediante técnicas de Deep Learning.

## Características

- Entrenamiento con YOLOv8.
- 32 clases de detección.
- Registro de experimentos mediante MLflow.
- Uso de GPU NVIDIA mediante CUDA.
- Exportación automática de pesos entrenados.

---

## Dataset

El dataset contiene las siguientes frutas:

- Apple
- Banana
- Grape
- Mango
- Melon
- Orange
- Peach
- Pear

Cada fruta posee cuatro estados:

- Unripe
- Ripe
- Overripe
- Rotten

Total:

- **8 frutas**
- **4 estados**
- **32 clases**

---

## Instalación

Crear un entorno de Conda

```bash
conda create -n agrovision python=3.11
conda activate agrovision
```

Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Entrenamiento

Ejemplo:

```bash
yolo detect train \
    data=dataset_processed/data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=128 \
    batch=8 \
    device=0 \
    workers=0
```

---

## Seguimiento de experimentos con MLflow

Habilitar el backend local

```bash
set MLFLOW_ALLOW_FILE_STORE=true
set MLFLOW_TRACKING_URI=mlruns
```

Abrir la interfaz

```bash
mlflow ui --backend-store-uri mlruns
```

Luego ingresar a

```
http://127.0.0.1:5000
```

---

## Tecnologías utilizadas

- Python
- PyTorch
- Ultralytics YOLOv8
- CUDA
- MLflow
- OpenCV
- NumPy
- Matplotlib

---

## Resultados

Durante el entrenamiento se registran automáticamente:

- Loss
- Precision
- Recall
- mAP@50
- mAP@50-95
- Curvas Precision-Recall
- Matriz de confusión

---

## Autor

Juan Camilo Torres