from fastapi import FastAPI


app = FastAPI(
    title="NASA Engine Failure Prediction API",
    version="1.0.0",
)


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "nasa-engine-failure-prediction"}
