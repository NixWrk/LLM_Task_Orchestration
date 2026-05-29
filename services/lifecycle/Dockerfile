FROM docker:27-cli AS docker-cli

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

COPY services/lifecycle/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orchestrator_core ./orchestrator_core
COPY services/lifecycle/lifecycle ./lifecycle

EXPOSE 4300

CMD ["uvicorn", "lifecycle.main:app", "--host", "0.0.0.0", "--port", "4300"]
