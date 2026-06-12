FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY src/docker/frontend/frontend_requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

EXPOSE 8501

CMD ["streamlit", "run", "src/docker/frontend/frontend_script.py", "--server.port=8501", "--server.address=0.0.0.0"]
