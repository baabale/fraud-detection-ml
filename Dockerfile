FROM tensorflow/tensorflow:2.10.0

WORKDIR /app

# Install Java for Spark and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jdk \
    curl \
    wget \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/default-java \
    CONFIG_PATH=/app/config.production.yaml \
    PYTHONPATH=/app

# Create a custom requirements file without TensorFlow (already installed)
RUN echo "# Core ML and Data Processing\n\
pyspark==3.3.0\n\
scikit-learn==1.1.3\n\
pandas==1.5.1\n\
numpy==1.23.4\n\
pyarrow==10.0.1\n\
fastparquet==0.8.3\n\
imbalanced-learn==0.10.1\n\
\n\
# Visualization\n\
matplotlib==3.6.2\n\
seaborn==0.12.1\n\
\n\
# Development and Notebooks\n\
jupyter==1.0.0\n\
\n\
# MLOps and Experiment Tracking\n\
mlflow==2.0.1\n\
pyyaml==6.0\n\
\n\
# Production Dependencies\n\
gunicorn==20.1.0\n\
flask==2.2.3\n\
redis==4.5.1\n\
prometheus-client==0.16.0\n\
prometheus-flask-exporter==0.22.3\n\
\n\
# Monitoring and Logging\n\
wandb==0.15.0" > requirements_docker.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy the entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data/raw /app/data/processed /app/data/test \
    /app/results/models /app/results/metrics /app/results/figures /app/results/monitoring

# Expose ports for services
EXPOSE 8000 5000

ENTRYPOINT ["docker-entrypoint.sh"]

# Default command (can be overridden at runtime)
CMD ["train"]
