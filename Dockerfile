FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY static/ ./static/
COPY models/ ./models/
COPY data/taxonomy.csv ./data/taxonomy.csv

ENV PORT=7860
EXPOSE 7860

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
