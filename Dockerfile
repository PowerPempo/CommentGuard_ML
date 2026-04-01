FROM python:3.13-slim

WORKDIR /app
ENV PYTHONPATH="/app"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["uvicorn", "src.models.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

