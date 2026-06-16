FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Copy and install frontend-specific requirements from the new path
COPY docker/frontend/frontend_requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# 2. Copy the central source code directory into the container
COPY src/ /app/src/

# 3. Set PYTHONPATH to ensure Python can resolve imports from the src directory
ENV PYTHONPATH="/app"

EXPOSE 8501

# 4. Run the UI dashboard directly from the centralized utils package
#CMD ["streamlit", "run", "src/utils/streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
CMD ["streamlit", "run", "src/utils/streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]