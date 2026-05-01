FROM python:3.10-slim

WORKDIR /app

COPY backend_ihs/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY metrics.py .
COPY backend_ihs/main.py .

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]