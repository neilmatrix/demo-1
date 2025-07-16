Here’s a clean **documentation block** with all the **build** and **run** commands for both the **MLflow server** and **training container**, using your `mlflow-net` network setup.

---

## ✅ 1. Create Docker Network (One-Time Setup)

```bash
docker network create mlflow-net
```

*(If it already exists, skip this step.)*

---

## ✅ 2. Build MLflow Server Image

Navigate to `practice/mlflow-server`:

```bash
cd practice/mlflow-server
docker build -t mlflow-server .
```

---

## ✅ 3. Run MLflow Server Container

```bash
docker run -d \
  --name mlflow-server \
  --network mlflow-net \
  -p 5050:5000 \
  -v $(pwd)/mlruns:/mlflow/mlruns \
  mlflow-server
```

* **MLflow UI**: [http://127.0.0.1:5050](http://127.0.0.1:5050)
* Data logs will persist in `mlflow-server/mlruns` (mounted volume).

---

## ✅ 4. Build Training Image

Navigate to `practice/training`:

```bash
cd practice/training
docker build -t mnist-training:latest .
```

---

## ✅ 5. Run Training Container

```bash
docker run --rm \
  --network mlflow-net \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  mnist-training:latest
```

> **Note:**
> If you already set `ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000` in your training Dockerfile, you can omit `-e MLFLOW_TRACKING_URI=...`.

---

### ✅ Optional: Override Parameters at Runtime

If your `MLproject` supports parameters, pass them like:

```bash
docker run --rm \
  --network mlflow-net \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  mnist-training:latest \
  python src/train.py --lr 0.01 --epochs 10 --batch_size 128
```

---

### ✅ Check Network Connections

```bash
docker network inspect mlflow-net --format '{{range .Containers}}{{.Name}} {{end}}'
```

---

🔥 Do you want me to also add:

* **Commands to stop/remove containers** quickly?
* **`mlflow run` command from host** so it uses this same setup?
* Or a **single script** that automates all these steps?
