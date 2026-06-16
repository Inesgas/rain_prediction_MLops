FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Copy and install prediction-specific requirements from the new path
COPY docker/prediction/prediction_requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# 2. Copy the centralized source code and models into the container
COPY src/ /app/src/
COPY models/ /app/models/

# 3. Set PYTHONPATH to allow clean module resolution from the src folder
ENV PYTHONPATH="/app"

# 4. Expose the new custom port 8081 to avoid local environment conflicts
EXPOSE 8081

# 5. Run the native python server via its main function on port 8081
CMD ["python", "-m", "src.models.api", "--host", "0.0.0.0", "--port", "8081"]
