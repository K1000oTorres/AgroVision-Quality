# 🍎 AgroVision Quality

**AgroVision Quality** es un MVP desarrollado para la detección automática del estado de maduración de frutas mediante técnicas de **Computer Vision** e **Inteligencia Artificial**.

El proyecto utiliza un modelo **YOLOv8** entrenado para identificar simultáneamente el tipo de fruta y su estado de maduración, exponiendo el modelo mediante una **API REST desarrollada con FastAPI** para facilitar su integración con aplicaciones web, móviles o sistemas IoT.

---

## Objetivo

Desarrollar un sistema capaz de automatizar el proceso de inspección visual de frutas mediante visión por computador, reduciendo la intervención manual y facilitando la clasificación de productos durante procesos de cosecha, selección o control de calidad.

---

## Características

* Detección automática mediante YOLOv8.
* Clasificación de **32 clases** (8 frutas × 4 estados de maduración).
* API REST desarrollada con FastAPI.
* Arquitectura modular siguiendo separación por capas.
* Registro de experimentos utilizando MLflow.
* Contenerización mediante Docker.
* Documentación automática con Swagger UI.

---

## Dataset

El modelo fue entrenado utilizando el conjunto de datos **Fruit Ripeness** publicado en **Roboflow Universe** por **Fruitectives Team**. El dataset contiene imágenes anotadas para detección de objetos y clasificación del estado de maduración de frutas. Está compuesto por **32 clases**, correspondientes a la combinación de ocho tipos de frutas y cuatro estados de maduración.

### Frutas incluidas

* Apple
* Banana
* Grape
* Mango
* Melon
* Orange
* Peach
* Pear

### Estados de maduración

* Unripe
* Ripe
* Overripe
* Rotten

### Organización del dataset

* **Train:** 88%
* **Validation:** 8%
* **Test:** 4%

El dataset fue exportado en formato **YOLOv8**, el cual fue utilizado para el entrenamiento del modelo de detección.

> **Nota:** Las imágenes originales no se incluyen en este repositorio. Solo se conserva la estructura del proyecto y las anotaciones necesarias para facilitar la reproducción del entrenamiento. Para obtener el conjunto de datos completo, descárgalo desde Roboflow Universe.

**Dataset original:**

https://universe.roboflow.com/fruitectives-team/fruit-ripeness-unjex/dataset/2

**Licencia:** CC BY 4.0


> **Nota:** Las imágenes originales del dataset no se incluyen en este repositorio. Solo se conserva la estructura del proyecto y las anotaciones para facilitar la reproducción del entrenamiento.

---

## Arquitectura

```text
AgroVision-Quality/
│
├── src/
│   ├── api/
│   ├── application/
│   ├── domain/
│   ├── infrastructure/
│   └── main.py
│
├── dataset_processed/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── models/
├── Dockerfile
├── docker-compose.yml
├── requirements-api.txt
├── requirements-dev.txt
└── README.md
```

---

## Tecnologías utilizadas

* Python 3.11
* FastAPI
* Uvicorn
* Ultralytics YOLOv8
* PyTorch
* OpenCV
* NumPy
* Pillow
* MLflow
* Docker
* Docker Compose

---

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/K1000oTorres/AgroVision-Quality.git

cd AgroVision-Quality
```

Crear el entorno virtual:

```bash
python -m venv .venv
```

Activar el entorno:

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements-api.txt
```

---

## Entrenamiento del modelo

Ejemplo de entrenamiento:

```bash
yolo detect train data=dataset_processed/data.yaml model=yolov8n.pt epochs=100 imgsz=128 batch=8 device=0 workers=0
```

---

## Ejecutar la API

```bash
uvicorn src.main:app --reload
```

La documentación interactiva estará disponible en:

```
http://127.0.0.1:8000/docs
```

---

## Uso con Docker

Construcción de la imagen:

```bash
docker compose up --build
```

La API quedará disponible en:

```
http://localhost:8000/docs
```

---

## Flujo del sistema

```text
Imagen

↓

FastAPI

↓

YOLOv8

↓

Predicción

↓

Respuesta JSON
```

Ejemplo de respuesta:

```json
{
  "status": "success",
  "quality_decision": "ACCEPT_BATCH",
  "summary": {
    "Apple Ripe": 2
  },
  "total_detections": 2
}
```

---

## Resultados

Durante el entrenamiento se obtienen automáticamente métricas como:

* Precision
* Recall
* mAP@50
* mAP@50-95
* Curvas Precision-Recall
* Matriz de confusión
* Loss de entrenamiento y validación

Los experimentos pueden visualizarse utilizando **MLflow**.

---

## Estado del proyecto

Proyecto desarrollado como **MVP funcional**, preparado para evolucionar hacia un sistema de inspección automática de frutas mediante servicios REST y despliegue en contenedores Docker.

---

## Autor

**Juan Camilo Torres**

GitHub: https://github.com/K1000oTorres
