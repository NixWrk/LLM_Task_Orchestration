FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY queue_proxy ./queue_proxy

EXPOSE 4100

CMD ["uvicorn", "queue_proxy.main:app", "--host", "0.0.0.0", "--port", "4100"]
