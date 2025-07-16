FROM python:3.9-slim

WORKDIR /mlflow

RUN pip install mlflow
RUN mkdir -p /mlflow/mlruns

# expose mlflow port
EXPOSE 5000

# start mlflow server when container starts
# CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000",
#      "--backend-store-uri", "/mlflow/mlruns",                   # store run metadata (params, metrics, tags)
#      "--default-artifact-root", "/mlflow/mlruns"]               # store models, plots, files

# single line to write the above command
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "/mlflow/mlruns", "--default-artifact-root", "/mlflow/mlruns"]