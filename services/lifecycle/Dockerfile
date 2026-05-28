FROM docker:27-cli AS docker-cli

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lifecycle ./lifecycle

EXPOSE 4300

CMD ["uvicorn", "lifecycle.main:app", "--host", "0.0.0.0", "--port", "4300"]
