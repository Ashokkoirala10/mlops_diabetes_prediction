import sys,os



BASE_DIR = "/home/ashok/mlops_project"
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)



from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

import json
import pickle
import pandas as pd
import redis
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine,text
import pymysql  # noqa: F401  # ensure driver is present
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,classification_report
)

from xgboost import XGBClassifier
from great_expectations.data_context import DataContext
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.exceptions import DataContextError

import matplotlib.pyplot as plt
import optuna
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import joblib
import shap
import matplotlib.pyplot as plt
import pickle



from monitoring.data_drift_monitor import monitor_data_drift
from monitoring.concept_drift_monitor import monitor_concept_drift



# shap path
SHAP_OUTPUT_DIR = os.path.join(BASE_DIR, "reports/shap")
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)


# ============= CONFIG ============
DB_CONN = "mysql+pymysql://ashok:Docker%4012345%21@localhost:3306/diabetes_ml"



# Tables
RAW_TABLE = "diabetes_raw"
TRANSFORMED_TABLE = "diabetes_features_16"

# Redis
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_DB = 0
RKEY_RAW = "diabetes:raw_df"
RKEY_TRANSFORMED = "diabetes:transformed_df"
RKEY_SELECTED_COLS = "diabetes:selected_cols"
RKEY_TUNED_MODEL_PATH = "diabetes:tuned_model_path"
RKEY_BEST_PARAMS = "diabetes:best_params"
RKEY_METRICS = "diabetes:metrics"
RKEY_TRAIN_TEST = "diabetes:train_test"

# Artifacts & tracking
DATA_DIR = f"{BASE_DIR}/data"
MODEL_DIR = f"{BASE_DIR}/models"
REPORT_DIR = f"{BASE_DIR}/reports"
ARROW_PATH = f"{DATA_DIR}/transformed_data.arrow"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
FEATURES_PATH = f"{MODEL_DIR}/selected_features.json"
BASELINE_MODEL_PATH = f"{MODEL_DIR}/xgb_model_baseline.pkl"
TUNED_MODEL_PATH = f"{MODEL_DIR}/xgb_model_tuned.pkl"
CONF_MAT_PATH = f"{MODEL_DIR}/confusion_matrix.png"
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "diabetes_mlops_experiment"

# File locations
CSV_PATH = f"{DATA_DIR}/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
GX_CONTEXT_DIR = f"{BASE_DIR}/gx"

# Ensure dirs exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Make project importable for monitoring module etc.
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ============= REDIS HELPERS ============
def get_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def df_to_bytes(df: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()

def bytes_to_df(b: bytes) -> pd.DataFrame:
    buf = pa.BufferReader(b)
    table = pq.read_table(buf)
    return table.to_pandas()


# ============= TASK FUNCS ============

def ingest_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)

    int_cols = [
        "Diabetes_binary","HighBP","HighChol","CholCheck","BMI","Smoker",
        "Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
        "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth",
        "MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"
    ]
    float_cols = ["BMI"]  

    for col in int_cols:
        df[col] = df[col].astype(int)
    
    for col in float_cols:
        df[col] = df[col].astype(float)

    # 1) drop duplicates
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows.")

    engine = create_engine(DB_CONN)
    df.to_sql(RAW_TABLE, con=engine, if_exists="replace", index=False)

    r = get_redis()
    r.set(RKEY_RAW, df_to_bytes(df))

    print(f" Ingested {df.shape[0]} rows into {RAW_TABLE} and cached to Redis key '{RKEY_RAW}'")


def validate_data():
    """
    Great Expectations validation for diabetes_raw table.
    Ensures datasource & suite exist, applies expectations, and logs details.
    """
    datasource_name = "diabetes_mysql"
    suite_name = "diabetes_raw_suite"

    try:
        context = DataContext(context_root_dir=GX_CONTEXT_DIR)
    except DataContextError as e:
        raise RuntimeError(f"Could not load Great Expectations context: {e}")

    # 1. Ensure datasource exists
    existing_sources = [ds["name"] for ds in context.list_datasources()]
    if datasource_name not in existing_sources:
        try:
            context.add_datasource(
                name=datasource_name,
                class_name="Datasource",
                execution_engine={
                    "class_name": "SqlAlchemyExecutionEngine",
                    "connection_string": DB_CONN
                },
                data_connectors={
                    "default_runtime_data_connector_name": {
                        "class_name": "RuntimeDataConnector",
                        "batch_identifiers": ["default_identifier_name"]
                    }
                }
            )
            print(f"Added datasource: {datasource_name}")
        except Exception as e:
            raise RuntimeError(f" Failed to add datasource '{datasource_name}': {e}")
    else:
        print(f"ℹ Datasource '{datasource_name}' already exists.")

    # 2. Ensure expectation suite exists
    try:
        context.get_expectation_suite(suite_name)
        print(f"ℹ Using existing expectation suite: {suite_name}")
    except DataContextError:
        # Updated for GE v0.14+: use add_expectation_suite instead of create_expectation_suite
        context.add_expectation_suite(suite_name)
        print(f" Created new expectation suite: {suite_name}")

    # 3. Batch request
    batch_request = RuntimeBatchRequest(
        datasource_name=datasource_name,
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name=RAW_TABLE,
        runtime_parameters={"query": f"SELECT * FROM {RAW_TABLE}"},
        batch_identifiers={"default_identifier_name": "validation_run"},
    )

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    # 4. Core expectations
    expected_columns = [
        "Diabetes_binary","HighBP","HighChol","CholCheck","BMI","Smoker",
        "Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
        "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth",
        "MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"
    ]
    validator.expect_table_columns_to_match_ordered_list(column_list=expected_columns)
    for col in expected_columns:
        validator.expect_column_values_to_not_be_null(col)

    numeric_expectations = {
        "BMI": (10, None),
        "MentHlth": (0, 30),
        "PhysHlth": (0, 30),
        "Age": (1, 13),
        "Education": (1, 6),
        "Income": (1, 8)
    }
    for col, (min_val, max_val) in numeric_expectations.items():
        validator.expect_column_values_to_be_between(col, min_value=min_val, max_value=max_val)


    # 5. Validate and report
    validator.save_expectation_suite(discard_failed_expectations=False)
    results = validator.validate()

    if results["success"]:
        print(" Data validation PASSED.")
        return {"status": "success", "checked_rows": results['statistics']['evaluated_expectations']}
    else:
        print(" Data validation FAILED. Issues found:")
        for res in results["results"]:
            if not res["success"]:
                expectation = res["expectation_config"]["expectation_type"]
                column = res["expectation_config"]["kwargs"].get("column", "TABLE")
                unexpected = res["result"].get("unexpected_count", 0)
                print(f" - {expectation} failed on {column}: {unexpected} unexpected values")
        raise ValueError("Data validation failed. Check GE Data Docs for full details.")

def transform_data():
    engine = create_engine(DB_CONN)
    r = get_redis()

    # Prefer Redis; fallback to DB
    raw_bytes = r.get(RKEY_RAW)
    if raw_bytes:
        df = bytes_to_df(raw_bytes)
    else:
        df = pd.read_sql(f"SELECT * FROM {RAW_TABLE}", con=engine)


    # scale numeric cols
    cols_to_scale = ["BMI", "MentHlth", "PhysHlth"]
    for c in cols_to_scale:
        if c not in df.columns:
            raise ValueError(f"Required column for scaling not found: {c}")
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    joblib.dump(scaler, SCALER_PATH)

    # select top 16 features (ANOVA F-score)
    if "Diabetes_binary" not in df.columns:
        raise ValueError("Target column 'Diabetes_binary' not found.")
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]

    k = min(16, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_cols = X.columns[selector.get_support()].tolist()

    # keep feature types as in ingestion (already cast)
    feature_types = {col: str(X[col].dtype) for col in selected_cols}


    # persist selected feature names and types
    features_dict = {
        "features": selected_cols,
        "types": feature_types
    }
    with open(FEATURES_PATH, "w") as f:
        json.dump(features_dict, f, indent=2)

    r.set(RKEY_SELECTED_COLS, pickle.dumps(selected_cols))
    r.set("diabetes_feature_types", pickle.dumps(feature_types))

    # rebuilt df with selected features + target
    df_selected = pd.concat([X[selected_cols], y], axis=1)

    #  save everywhere
    with pa.OSFile(ARROW_PATH, "wb") as f:
        pq.write_table(pa.Table.from_pandas(df_selected), f)
    r.set(RKEY_TRANSFORMED, df_to_bytes(df_selected))
    df_selected.to_sql(TRANSFORMED_TABLE, con=engine, if_exists="replace", index=False)

    print(f" Transform complete: {df_selected.shape} → SQL({TRANSFORMED_TABLE}), Redis({RKEY_TRANSFORMED}), Arrow({ARROW_PATH})")
    print(f" Selected features ({len(selected_cols)}): {selected_cols}")
    print(f" Feature types: {feature_types}")


def train_initial_model():
    """Train a baseline XGB model with default-ish params on transformed data"""
    r = get_redis()
    tbytes = r.get(RKEY_TRANSFORMED)
    if tbytes:
        df = bytes_to_df(tbytes)
    else:
        df = pd.read_sql(f"SELECT * FROM {TRANSFORMED_TABLE}", con=create_engine(DB_CONN))

    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline_model = XGBClassifier(
        # xgboost>=1.7 no need for use_label_encoder
        eval_metric="logloss",
        random_state=42
    )
    baseline_model.fit(X_train, y_train)
    y_pred = baseline_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    # Save baseline model
    joblib.dump(baseline_model, BASELINE_MODEL_PATH)
    print(f" Baseline trained. Acc={accuracy:.3f}, F1={f1_macro:.3f}")
    return {"accuracy": accuracy, "f1_macro": f1_macro, "model_path": BASELINE_MODEL_PATH}


def hypertune_model():
    """Run Optuna hyperparameter tuning; save best artifacts + data to Redis + disk"""
    r = get_redis()
    tbytes = r.get(RKEY_TRANSFORMED)
    if tbytes:
        df = bytes_to_df(tbytes)
    else:
        df = pd.read_sql(f"SELECT * FROM {TRANSFORMED_TABLE}", con=create_engine(DB_CONN))

    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "eval_metric": "logloss",
            "random_state": 42
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return f1_score(y_test, preds, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params

    # Train final model with best params
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"Hypertuned model. Acc={accuracy:.3f}, F1={f1_macro:.3f}")

    # Save tuned model
    joblib.dump(final_model, TUNED_MODEL_PATH)

    # Save everything in Redis
    r.set(RKEY_TUNED_MODEL_PATH, TUNED_MODEL_PATH.encode())
    r.set(RKEY_BEST_PARAMS, pickle.dumps(best_params))
    r.set(RKEY_METRICS, pickle.dumps({"accuracy": accuracy, "f1_macro": f1_macro}))
    r.set(RKEY_TRAIN_TEST, pickle.dumps({
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test
    }))
    return "Hypertuning complete and saved to Redis"


def generate_shap_plots_from_redis():
    """Compute SHAP values using the trained model and test data from Redis (reproducible)."""
    r = get_redis()

    # Load latest tuned model
    tuned_path_b = r.get(RKEY_TUNED_MODEL_PATH)
    if not tuned_path_b:
        raise RuntimeError("Tuned model not found in Redis")
    model_path = tuned_path_b.decode()
    model = joblib.load(model_path)

    # Load train/test split
    train_test_b = r.get(RKEY_TRAIN_TEST)
    if not train_test_b:
        raise RuntimeError("Train/test data not found in Redis")
    data = pickle.loads(train_test_b)
    X_train, X_test = data["X_train"], data["X_test"]

    # SHAP Explainer
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)  # returns Explanation object

    # Global summary plot
    summary_path = os.path.join(SHAP_OUTPUT_DIR, "shap_summary.png")
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()

    # Local explanations (first 5 samples)
    local_paths = []
    for i in range(min(5, len(X_test))):
        local_path = os.path.join(SHAP_OUTPUT_DIR, f"shap_local_{i}.png")
        plt.figure()
        shap.plots.waterfall(shap_values[i], show=False)
        plt.savefig(local_path, bbox_inches="tight")
        plt.close()
        local_paths.append(local_path)

    print(f"SHAP plots saved to {SHAP_OUTPUT_DIR}")
    return summary_path, local_paths


def log_model_mlflow():
    """Log the trained hyperparameter-tuned model + artifacts to MLflow and promote it to Production"""
    r = get_redis()
    tuned_path_b = r.get(RKEY_TUNED_MODEL_PATH)
    best_params_b = r.get(RKEY_BEST_PARAMS)
    metrics_b = r.get(RKEY_METRICS)
    data_b = r.get(RKEY_TRAIN_TEST)

    if not all([tuned_path_b, best_params_b, metrics_b, data_b]):
        raise RuntimeError("Missing tuned model artifacts in Redis; ensure previous tasks succeeded.")

    tuned_path = tuned_path_b.decode()
    best_params = pickle.loads(best_params_b)
    metrics = pickle.loads(metrics_b)
    data = pickle.loads(data_b)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    model = joblib.load(tuned_path)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Confusion matrix (test set)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_test_pred),
        display_labels=["No Diabetes", "Diabetes"]
    )
    disp.plot()
    plt.title("Confusion Matrix (Test)")
    plt.savefig(CONF_MAT_PATH)
    plt.close()

    # Classification reports
    train_class_report = classification_report(
        y_train,
        y_train_pred,
        target_names=["No Diabetes", "Diabetes"],
        output_dict=True
    )
    test_class_report = classification_report(
        y_test,
        y_test_pred,
        target_names=["No Diabetes", "Diabetes"],
        output_dict=True
    )

    # Load selected features + types
    features_dict = None
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as f:
            features_dict = json.load(f)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"xgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        # Log hyperparameters
        mlflow.log_params(best_params)

        # Log main metrics
        mlflow.log_metric("accuracy", float(metrics["accuracy"]))
        mlflow.log_metric("f1_macro", float(metrics["f1_macro"]))

        # Log per-class metrics (test)
        for label, scores in test_class_report.items():
            if isinstance(scores, dict):
                for metric_name in ["precision", "recall", "f1-score"]:
                    mlflow.log_metric(f"test_{label}_{metric_name}", float(scores[metric_name]))

        # Log per-class metrics (train)
        for label, scores in train_class_report.items():
            if isinstance(scores, dict):
                for metric_name in ["precision", "recall", "f1-score"]:
                    mlflow.log_metric(f"train_{label}_{metric_name}", float(scores[metric_name]))

        # Log artifacts
        mlflow.log_dict(test_class_report, artifact_file="classification_report_test.json")
        mlflow.log_dict(train_class_report, artifact_file="classification_report_train.json")
        if features_dict:
            mlflow.log_dict(features_dict, artifact_file="features/selected_features.json")

        if os.path.exists(tuned_path):
            mlflow.log_artifact(tuned_path)
        if os.path.exists(SCALER_PATH):
            mlflow.log_artifact(SCALER_PATH)
        if os.path.exists(CONF_MAT_PATH):
            mlflow.log_artifact(CONF_MAT_PATH)
        if os.path.exists(ARROW_PATH):
            mlflow.log_artifact(ARROW_PATH)

        # Log and register model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="xgb_model",
            registered_model_name="Diabetes_XGB_Model"
        )

        # Automatically promote to Production
        client = MlflowClient()
        model_name = "Diabetes_XGB_Model"
        latest_version_info = client.get_latest_versions(name=model_name, stages=["None"])[0]  # pick latest version
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version_info.version,
            stage="Production",
            archive_existing_versions=True
        )

    print(f"Logged tuned model, classification reports, selected features, and promoted version {latest_version_info.version} to Production")

TABLE_NAME = "predictions_auto"
FEATURES_PATH = os.path.join(BASE_DIR, "models/selected_features.json")

# Load feature columns & types
def create_predictions_table():
    """Create predictions_auto table if not exists"""


    engine = create_engine(DB_CONN)  # define engine inside function

    # Load feature columns & types at runtime
    with open(FEATURES_PATH) as f:
        features_info = json.load(f)
    feature_cols = features_info["features"]
    feature_types = features_info["types"]

    cols_sql = []
    for col in feature_cols:
        col_type = "FLOAT" if feature_types[col] == "float" else "TINYINT"
        cols_sql.append(f"`{col}` {col_type}")
    cols_sql = ",\n".join(cols_sql)

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        {cols_sql},
        prediction TINYINT,
        probability FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))
    print(f" Table '{TABLE_NAME}' is ready.")


# ============= AIRFLOW DAG ============

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": True,
    "email": ["koiralaashok02@gmail.com"],  # recipients here
}

dag = DAG(
    dag_id="mlops_diabetes_pipeline",
    default_args=default_args,
    description="End-to-end MLOps pipeline: ingest → validate → transform → train → tune → MLflow → monitor with failure alert",
    start_date=datetime(2025, 8, 20),
    schedule_interval="@monthly",
    catchup=False,
)

activate_env = BashOperator(
    task_id='activate_env',
    bash_command='conda run -n mysecondenvironment /bin/bash -c "echo Environment activated"',
    dag=dag
)

#  Start Redis container
start_redis = BashOperator(
    task_id='start_redis',
    bash_command='docker start my-redis || docker run -d --name my-redis -p 6379:6379 redis:latest', #chnage you container name accordingly(my-redis)
    dag=dag
)

#  Start Docker container for MariaDB
start_docker = BashOperator(
    task_id='start_docker',
    bash_command='docker start mcs_container || docker run -d --name mcs_container -e MYSQL_ROOT_PASSWORD=root -p 3306:3306 mariadb:latest',#chnage you container name accordingly(mcs_container)
    dag=dag
)

#  Test database connection with your credentials
docker_exec = BashOperator(
    task_id='mariadb_conn_testing',
    bash_command='docker exec -i mcs_container mariadb -u ashok -p"Docker@12345!" -h 127.0.0.1 -P 3306 diabetes_ml -e "SELECT 1;"', #chnage you container name accordingly(mcs_container and username and password)
    dag=dag
)

start_mlflow = BashOperator(
    task_id='start_mlflow',
    bash_command="""
    cd /home/ashok/mlops_project 
    nohup conda run -n mysecondenvironment mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000 > /tmp/mlflow.log 2>&1 < /dev/null &
    sleep 5
    if curl -s http://127.0.0.1:5000 | grep -q MLflow; then
        echo "MLflow started successfully"
    else
        echo "MLflow failed to start"
        cat /tmp/mlflow.log
        exit 1
    fi
    """, #change the cd as of yours (/home/ashok/mlops_project )
    dag=dag
)
# Start FastAPI app with uvicorn
start_fastapi = BashOperator(
    task_id="start_fastapi",
    bash_command="""
    cd /home/ashok/mlops_project
    nohup conda run -n mysecondenvironment uvicorn app:app --reload --host 0.0.0.0 --port 8000 > /tmp/fastapi.log 2>&1 < /dev/null &
    sleep 5
    if curl -s http://127.0.0.1:8000/docs | grep -q "Swagger"; then
        echo "FastAPI started successfully"
    else
        echo "FastAPI failed to start"
        cat /tmp/fastapi.log
        exit 1
    fi
    """, #change the cd as of yours (/home/ashok/mlops_project )
    dag=dag
)

# Start Streamlit app
start_streamlit = BashOperator(
    task_id="start_streamlit",
    bash_command="""
    cd /home/ashok/mlops_project
    nohup conda run -n mysecondenvironment streamlit run streamlit_app.py --server.port 8501 > /tmp/streamlit.log 2>&1 < /dev/null &
    sleep 5
    if curl -s http://127.0.0.1:8501 | grep -q "<title>Streamlit"; then
        echo "Streamlit started successfully"
    else
        echo "Streamlit failed to start"
        cat /tmp/streamlit.log
        exit 1
    fi
    """, #change the cd as of yours(/home/ashok/mlops_project )
    dag=dag
)


ingest_task = PythonOperator(
    task_id="ingest_data",
    python_callable=ingest_data,
    dag=dag
)

validate_task = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id="transform_data",
    python_callable=transform_data,
    dag=dag
)

baseline_task = PythonOperator(
    task_id="train_initial_model",
    python_callable=train_initial_model,
    dag=dag
)

hypertune_task = PythonOperator(
    task_id="hypertune_model",
    python_callable=hypertune_model,
    dag=dag
)
shap_task = PythonOperator(
    task_id="generate_shap_explanations",
    python_callable=generate_shap_plots_from_redis,
    dag=dag
)

mlflow_task = PythonOperator(
    task_id="log_model_mlflow",
    python_callable=log_model_mlflow,
    dag=dag
)
create_table_task = PythonOperator(
    task_id="create_predictions_table",
    python_callable=create_predictions_table,
    dag=dag
)
data_drift_task = PythonOperator(
    task_id='data_drift',
    python_callable=monitor_data_drift
)

concept_drift_task = PythonOperator(
    task_id='concept_drift',
    python_callable=monitor_concept_drift
)

# Dependencies
activate_env >> start_redis >> start_docker >> docker_exec >> start_mlflow >> ingest_task >> validate_task >> transform_task >> baseline_task >> hypertune_task >> shap_task >> mlflow_task >> create_table_task >> data_drift_task >> concept_drift_task >> start_fastapi >> start_streamlit


