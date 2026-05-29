FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/healthcheck/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orchestrator_core ./orchestrator_core
COPY services/healthcheck/app ./app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
