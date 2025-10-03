import streamlit as st
import json
import os
from glob import glob
import pandas as pd
import plotly.express as px
from collections.abc import Mapping
from PIL import Image

# ---------- CONFIG ----------
BASE_REPORT_DIR = "/home/ashok/mlops_project/reports/evidently/" #chnage this with your path
SHAP_REPORT_DIR = "/home/ashok/mlops_project/reports/shap/" #chnage this with your path
DATA_DRIFT_DIR = os.path.join(BASE_REPORT_DIR, "data_drift")
CONCEPT_DRIFT_DIR = os.path.join(BASE_REPORT_DIR, "concept_drift")

st.set_page_config(page_title="ML Model Monitoring Dashboard", layout="wide")
st.title("ML Model Monitoring Dashboard")
st.markdown("View Evidently Data Drift & Concept Drift Reports, SHAP Explanations, and Predictor UI")

# ---------- Helper to flatten nested dicts ----------
def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# ---------- Helper to get latest HTML report ----------
def get_latest_html_report(folder):
    html_files = sorted(glob(os.path.join(folder, "*.html")), reverse=True)
    if not html_files:
        return None
    return html_files[0]

# ---------- Tabs ----------
tabs = st.tabs(["Data Drift", "Concept Drift", "SHAP Explanations", "Diabetes Predictor"])

# ----- Data Drift & Concept Drift -----
for i, tab_name in enumerate(["Data Drift", "Concept Drift"]):
    with tabs[i]:
        folder = DATA_DRIFT_DIR if tab_name == "Data Drift" else CONCEPT_DRIFT_DIR
        html_file = get_latest_html_report(folder)

        if not html_file:
            st.warning(f"No {tab_name} reports found!")
            continue

        st.subheader(f"Latest {tab_name} Report")
        st.write(f"HTML: `{os.path.basename(html_file)}`")

        # Display HTML report
        st.markdown("### HTML Report")
        with open(html_file, "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=1200, scrolling=True)

        st.success(f"{tab_name} report loaded successfully!")

        # Optional: load JSON metrics
        json_file = html_file.replace(".html", ".json")
        if os.path.exists(json_file):
            st.markdown("### JSON Summary")
            with open(json_file, "r") as f:
                data = json.load(f)

            metrics = data.get("metrics", [])
            drift_detected = any(
                m.get("result", {}).get("dataset_drift") for m in metrics
            )
            if drift_detected:
                st.error(f"{tab_name} Detected!")
            else:
                st.success(f"No {tab_name} Detected")

            summary_list = []
            for metric in metrics:
                metric_name = metric.get("metric_name", "N/A")
                result = metric.get("result", {})
                flat_result = flatten_dict(result)
                for key, value in flat_result.items():
                    if isinstance(value, (int, float, bool)):
                        summary_list.append({"Metric": metric_name, "Name": key, "Value": value})

            if summary_list:
                df_summary = pd.DataFrame(summary_list)
                st.markdown("### Metrics Visualization")
                fig = px.bar(
                    df_summary,
                    x="Name",
                    y="Value",
                    color="Metric",
                    barmode="group",
                    title=f"{tab_name} Metrics Summary"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No JSON metrics found for this report.")

# ----- SHAP Explanations -----

with tabs[2]:
    st.subheader("SHAP Explanations")
    
    if not os.path.exists(SHAP_REPORT_DIR):
        st.warning("No SHAP reports found. Please run DAG to generate SHAP plots.")
    else:
        # Global summary
        summary_path = os.path.join(SHAP_REPORT_DIR, "shap_summary.png")
        if os.path.exists(summary_path):
            st.markdown("### Global Feature Importance")
            img = Image.open(summary_path)
            st.image(img, width=900)  # <-- Set width in pixels (adjust as needed)
        else:
            st.info("Global SHAP summary plot not found.")
        
        # Local explanations
        local_files = sorted(glob(os.path.join(SHAP_REPORT_DIR, "shap_local_*.png")))
        if local_files:
            st.markdown("### Local Explanations (first 5 samples)")
            for lf in local_files:
                st.image(Image.open(lf), width=800)  # <-- Smaller width for local plots
        else:
            st.info("Local SHAP plots not found.")


# ----- Predictor UI -----
with tabs[3]:
    st.subheader("Interactive Predictor UI")
    with open("frontend/index.html", "r") as f:   # save your HTML file
        predictor_html = f.read()
    st.components.v1.html(predictor_html, height=1200, scrolling=True)
