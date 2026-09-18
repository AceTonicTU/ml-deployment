from fastapi import FastAPI

app = FastAPI(
    title="Vision Inference Service",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok!",
        "model_loaded": False,
    }