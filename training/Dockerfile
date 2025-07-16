FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y git && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the training code (including MLproject)
COPY . .

# Set MLflow to point to the MLflow server container (hostname will be container name)
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
# ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5050
# ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5050



# Optional: default command, if you want
CMD ["python", "src/train.py", "--lr", "0.001", "--epochs", "5", "--batch_size", "64"]