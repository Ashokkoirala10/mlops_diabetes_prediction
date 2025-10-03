import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import CatTargetDriftTab
from evidently import ColumnMapping
from sqlalchemy import create_engine
from pathlib import Path

# ---------- CONFIG ----------
DB_URL = "mysql+pymysql://ashok:Docker%4012345%21@localhost:3306/diabetes_ml"
TRANSFORMED_TABLE = "diabetes_features_16"  # Reference features (training data)
PREDICTIONS_TABLE = "predictions_auto"      # Current predictions
TARGET_COL = "Diabetes_binary"              # Original target column in training data
REPORT_DIR = Path("/home/ashok/mlops_project/reports/evidently/concept_drift")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MAX_ROWS = 1000  # Max rows to sample from reference and current

engine = create_engine(DB_URL)

def monitor_concept_drift():
    # Load datasets
    ref_df = pd.read_sql(f"SELECT * FROM {TRANSFORMED_TABLE}", con=engine)
    cur_df = pd.read_sql(f"SELECT * FROM {PREDICTIONS_TABLE}", con=engine)

    # Check for empty tables
    if ref_df.empty:
        print("Reference dataset is empty. Cannot compute concept drift.")
        return
    if cur_df.empty:
        print("Current predictions dataset is empty. Skipping concept drift monitoring.")
        return

    # Sample reference and current datasets if too large
    if len(ref_df) > MAX_ROWS:
        ref_df = ref_df.sample(n=MAX_ROWS, random_state=42)

    if len(cur_df) > MAX_ROWS:
        if "created_at" in cur_df.columns:
            cur_df = cur_df.sort_values("created_at", ascending=False).head(MAX_ROWS)
        else:
            cur_df = cur_df.sample(n=MAX_ROWS, random_state=42)

    # Column mapping for Evidently
    column_mapping = ColumnMapping()
    column_mapping.target = TARGET_COL

    # Align columns
    feature_cols = [col for col in ref_df.columns if col != TARGET_COL]
    ref_data = ref_df[feature_cols + [TARGET_COL]]

    cur_data = cur_df[feature_cols + ["prediction"]].copy()
    cur_data.rename(columns={"prediction": TARGET_COL}, inplace=True)  # Fix column name


    # Create dashboard and calculate
    dashboard = Dashboard(tabs=[CatTargetDriftTab()])
    dashboard.calculate(reference_data=ref_data, current_data=cur_data, column_mapping=column_mapping)

    # Save report
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"concept_drift_report_{ts}.html"
    dashboard.save(report_path)

    print(f"Concept drift report saved at {report_path}")
    return str(report_path)
