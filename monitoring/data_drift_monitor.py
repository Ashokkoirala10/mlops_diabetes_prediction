import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from sqlalchemy import create_engine
from pathlib import Path
from email.message import EmailMessage
import smtplib
import json
import os

# ---------- CONFIG ----------
DB_URL = "mysql+pymysql://ashok:Docker%4012345%21@localhost:3306/diabetes_ml"
TRANSFORMED_TABLE = "diabetes_features_16"
PREDICTIONS_TABLE = "predictions_auto"
REPORT_DIR = Path("/home/ashok/mlops_project/reports/evidently/data_drift")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MAX_ROWS = 1000

# Email Configuration (use environment variables in production)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "koiralaashok47@gmail.com") #put your mail
APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "bhkkffwntbakqccn") # put your app password
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "koiralaashok02@gmail.com") # put the receiptant password

# Create DB engine
engine = create_engine(DB_URL)

def monitor_data_drift():
    try:
        ref_df = pd.read_sql(f"SELECT * FROM {TRANSFORMED_TABLE}", con=engine)
        cur_df = pd.read_sql(f"SELECT * FROM {PREDICTIONS_TABLE}", con=engine)
    except Exception as e:
        print(f" Failed to fetch data from database: {e}")
        return

    if ref_df.empty:
        print("Reference dataset is empty. Cannot compute data drift.")
        return
    if cur_df.empty:
        print("Current predictions dataset is empty. Skipping data drift monitoring.")
        return

    # Sample datasets
    if len(ref_df) > MAX_ROWS:
        ref_df = ref_df.sample(n=MAX_ROWS, random_state=42)
    if len(cur_df) > MAX_ROWS:
        if "created_at" in cur_df.columns:
            cur_df = cur_df.sort_values("created_at", ascending=False).head(MAX_ROWS)
        else:
            cur_df = cur_df.sample(n=MAX_ROWS, random_state=42)

    # Align feature columns
    feature_cols = [col for col in ref_df.columns if col != "Diabetes_binary"]
    ref_data = ref_df[feature_cols]
    cur_data = cur_df[feature_cols]

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"data_drift_report_{ts}.html"

    dashboard = Dashboard(tabs=[DataDriftTab()])
    dashboard.calculate(ref_data, cur_data)
    dashboard.save(report_path)
    print(f" Data drift report saved at {report_path}")

    # Programmatic drift detection
    profile = Profile(sections=[DataDriftProfileSection()])
    profile.calculate(ref_data, cur_data)
    profile_data = json.loads(profile.json())

    drift_detected = False
    metrics_dict = profile_data.get("data_drift", {}).get("data", {}).get("metrics", {})
    for feature_name, metrics in metrics_dict.items():
        if isinstance(metrics, dict) and metrics.get("drift_detected", False):
            print(f" Drift detected in feature: {feature_name}")
            drift_detected = True

    if not drift_detected:
        print(" No data drift detected.")
    else:
        print("Data drift detected! Sending email alert...")
        send_email_alert(report_path)

    return drift_detected

def send_email_alert(report_path):
    subject = " Data Drift Detected!"
    body = f"""
    Data drift has been detected in your system.

    Drift report location: {report_path}

    Please review the report for details.
    """

    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f" Failed to send email: {e}")

if __name__ == "__main__":
    monitor_data_drift()
