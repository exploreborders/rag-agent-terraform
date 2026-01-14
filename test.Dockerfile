FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy test app
COPY test_app.py .

# Install compatible versions
RUN pip install fastapi==0.104.1 starlette==0.27.0 uvicorn prometheus-client starlette-exporter

EXPOSE 8000

CMD ["uvicorn", "test_app:app", "--host", "0.0.0.0", "--port", "8000"]