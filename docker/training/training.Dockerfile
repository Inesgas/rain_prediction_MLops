FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Copy and install training-specific requirements from the new path
COPY docker/training/training_requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# 2. Copy the necessary directories into the container context
COPY src/ /app/src/
COPY data/ /app/data/
COPY models/ /app/models/
COPY references/ /app/references/

# 3. Set PYTHONPATH to ensure Python can resolve imports from the src package
ENV PYTHONPATH="/app"

# 4. Default training module to run (can be overridden at runtime)
ENV TRAIN_MODULE="src.models.train_winner"

# 5. Run the training script dynamically as a module
CMD ["sh", "-c", "python -m ${TRAIN_MODULE}"]
