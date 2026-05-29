FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/queue_proxy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orchestrator_core ./orchestrator_core
COPY services/queue_proxy/queue_proxy ./queue_proxy

EXPOSE 4100

CMD ["uvicorn", "queue_proxy.main:app", "--host", "0.0.0.0", "--port", "4100"]
