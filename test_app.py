from fastapi import FastAPI
from starlette_exporter import PrometheusMiddleware, handle_metrics

app = FastAPI()
app.add_middleware(PrometheusMiddleware, app_name="test", prefix="test")
app.add_route("/metrics", handle_metrics)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def health():
    return {"status": "healthy"}
