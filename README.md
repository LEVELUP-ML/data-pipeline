# Data Pipeline

End-to-end MLOps pipeline for data ingestion -> processing -> training -> evaluation -> orchestration with Airflow -> tracking with MLflow.

## 1) Project Structure

```
.
├── config/                 # Config files (Airflow, app settings, etc.)
├── dags/                   # Airflow DAGs (if used)
├── data/
├── logs/                   # Runtime logs (not committed)
├── docker-compose.yaml     # Local services (Airflow/Postgres/Redis/etc.)
├── Dockerfile              
├── requirements.txt
└── README.md
```

## 2) Requirements

- Python 3.11+
- Docker + Docker Compose

## 3) Setup

### 3.1 Environment variables

Create a `.env` file, with a Kaggle Account API (do not commit):

```bash
KAGGLE_USERNAME=
KAGGLE_KEY=
```

## 4) Data

### 4.1 Where to put data

* Put raw data under: `data/raw/`
* Generated processed data goes to: `data/processed/`

### 4.2 Data is not committed

This repo does not commit datasets or large artifacts to Git.


## 5) Running with Docker Compose (Airflow Local)

### 5.1 Bring up services

```bash
docker compose up -d
```

Check status:

```bash
docker compose ps
```

### 5.2 Airflow UI

* URL: [http://localhost:8080](http://localhost:8080)
* Username/password: `admin` / `admin` (default for local dev)

### 5.3 Useful debugging commands

```bash
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-apiserver
docker compose logs -f postgres
```

### 5.4 Reset local state (dev only)

```bash
docker compose down
# Nuclear — also removes volumes (deletes DB, logs)
docker compose down -v
# Start back up
docker compose up -d
```

## 6) What’s Next (Planned)

* [ ] Add testing/monitoring code