from fastapi import FastAPI

from .api.routes import router

app = FastAPI(
    title="AgroVision Quality API",
    description="MVP para deteccion de madurez y condicion de frutas usando YOLO.",
    version="0.1.0",
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "agrovision-quality-api",
    }
