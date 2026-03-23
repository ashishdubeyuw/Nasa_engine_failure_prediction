from fastapi import FastAPI


app = FastAPI(
    title="NASA Engine Failure Prediction API",
    version="1.0.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"status": "ok", "service": "nasa-engine-failure-prediction"}
