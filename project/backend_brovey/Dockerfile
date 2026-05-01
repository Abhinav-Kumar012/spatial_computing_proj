FROM python:3.10-slim

WORKDIR /app
COPY backend_brovey/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source scripts
COPY metrics.py .
COPY backend_brovey/main.py .

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
