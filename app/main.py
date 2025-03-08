from fastapi import FastAPI
from .routers import prediction

app = FastAPI(
    title="ML Prediction API",
    description="API for making ML predictions",
    version="0.1.0"
)

app.include_router(prediction.router, tags=["Prediction"])


@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Prediction API"}
