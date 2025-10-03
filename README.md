🚀 MLOps Project – Diabetes Prediction
📌 Overview

This project demonstrates a production-ready MLOps workflow for training, validating, deploying, and monitoring a machine learning model for Diabetes Prediction.

It integrates:

Conda for environment management

MariaDB ColumnStore + Redis for data & caching

Airflow for orchestration

Great Expectations for data quality validation

MLflow for experiment tracking

Evidently AI for monitoring data drift & concept drift

FastAPI and Streamlit for serving predictions

🛠️ Prerequisites

Clone this repository

git clone https://github.com/Ashokkoirala10/mlops_diabetes_prediction.git
cd mlops_diabetes_prediction/mlops_project


Install Docker (for DB & Redis)

# Pull MariaDB ColumnStore
docker pull mariadb/columnstore

# Pull Redis
docker pull redis


Install Conda (Miniconda or Anaconda)

⚙️ Setup Environment
conda env create -f environment.yml
conda activate mlops_env

🌬️ Airflow Setup

Initialize Airflow DB:

airflow db init


Create Admin user:

airflow users create \
  --username admin \
  --firstname First \
  --lastname Last \
  --role Admin \
  --email admin@example.com \
  --password admin

⚠️ Note

Update hardcoded paths in the DAG file where indicated by comments.

Example:

BASE_DIR = "/home/ashok/mlops_project"  # Change to your absolute path
DB_CONN = "mysql+pymysql://<username>:<password>@localhost:3306/diabetes_ml"

✅ Great Expectations Setup

If not initialized:

great_expectations init

📊 Monitoring (Data Drift & Concept Drift)

In monitoring DAG/config files, update:

DB_URL = "mysql+pymysql://<username>:<password>@localhost:3306/diabetes_ml"

REPORT_DIR = Path("/your/path/mlops_project/reports/evidently/concept_drift")
REPORT_DIR = Path("/your/path/mlops_project/reports/evidently/data_drift")

⚡ FastAPI Setup

Update DB connection credentials with your username/password.

Update directory paths as per your system.

Run FastAPI:

uvicorn app:app --reload


Access: http://127.0.0.1:8000/docs

🎨 Streamlit Setup

In streamlit_app.py, update the file paths (comments provided in code).

Run Streamlit:

streamlit run streamlit_app.py

📊 MLflow Tracking

Start MLflow UI:

mlflow ui


Access MLflow: http://127.0.0.1:5000

📈 Features

Airflow DAGs orchestrating ingestion, validation, training, monitoring

Great Expectations for schema & data checks

Evidently reports for monitoring data/concept drift

FastAPI for REST API prediction service

Streamlit for interactive predictions

MLflow for experiment tracking

👤 Author

Ashok Koirala