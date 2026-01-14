# Minimal test Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install just the basics for testing
RUN pip install fastapi==0.104.1 uvicorn==0.24.0 starlette==0.27.0

COPY src/app/main.py .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]