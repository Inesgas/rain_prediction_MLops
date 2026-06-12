FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY src/docker/prediction/prediction_requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY models/ /app/models/

EXPOSE 8000

CMD ["python", "-m", "src.docker.prediction.prediction_api", "--host", "0.0.0.0", "--port", "8000"]
