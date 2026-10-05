FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    mlflow==2.22.0 \
    scikit-learn==1.6.1 \
    numpy==1.26.4 \
    pandas==2.0.3 \
    boto3 \
    psycopg2-binary

COPY main.py .

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]