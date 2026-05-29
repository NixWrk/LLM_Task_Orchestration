FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/fake_openai_backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orchestrator_core ./orchestrator_core
COPY services/fake_openai_backend/fake_openai_backend ./fake_openai_backend

EXPOSE 18080

CMD ["uvicorn", "fake_openai_backend.main:app", "--host", "0.0.0.0", "--port", "18080"]
