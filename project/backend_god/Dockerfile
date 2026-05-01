# Use specific pytorch version to ensure CUDA compatibility and prevent huge pip downloads
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY backend_god/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source scripts. We mount context in docker-compose as ./project so we access root files like fusion_model.py
COPY fusion_model.py .
COPY metrics.py .
COPY backend_god/main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
