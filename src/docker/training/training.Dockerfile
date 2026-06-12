FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY src/docker/training/training_requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY data/ /app/data/
COPY models/ /app/models/
COPY references/ /app/references/

CMD ["python", "-m", "src.docker.training.training_script"]
