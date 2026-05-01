# Builder stage
FROM python:3.10-slim AS builder

WORKDIR /build
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Runtime stage
FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY frontend/app.py .

# Expose gradio port
EXPOSE 7860
CMD ["python", "app.py"]
