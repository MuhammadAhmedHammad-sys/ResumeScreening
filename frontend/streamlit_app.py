from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st

# -----------------------------------------
# CONFIG & SETUP
# -----------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (BASE_DIR / ".." / "data" / "skillset.json").resolve()
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Upgraded Page Config
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

def load_skills() -> List[str]:
    if not DATA_PATH.exists():
        st.warning(f"Skill file not found: {DATA_PATH}. Please add data/skillset.json.")
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        skills = json.load(f)
    return sorted([str(s).strip() for s in skills if str(s).strip()])

all_skills = load_skills()

# -----------------------------------------
# SIDEBAR
# -----------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/942/942748.png", width=80) # Adds a nice icon
    st.header("System Config")
    api_url = st.text_input("FastAPI Base URL", value=DEFAULT_API_URL)
    st.caption("Ensure your FastAPI backend is running.")
    st.divider()
    st.write("### Instructions")
    st.write("1. Add candidate profiles.")
    st.write("2. Define the job requirements.")
    st.write("3. Hit Predict to rank the batch.")

# -----------------------------------------
# MAIN HEADER
# -----------------------------------------
st.title("📄 AI Resume Screening Pipeline")
st.markdown("Automate candidate ranking and decision-making using Machine Learning.")
st.divider()

if "rows" not in st.session_state:
    st.session_state.rows = [
        {"candidate_id": "Candidate 1", "resume_skills": [], "job_skills": [], "resume_years": 0, "job_years": 0}
    ]

# -----------------------------------------
# CANDIDATE INPUT SECTION
# -----------------------------------------
st.markdown("### 👥 Candidate Batch Entry")

# Layout for action buttons
col_a, col_b, _ = st.columns([1, 1, 4])
with col_a:
    if st.button("➕ Add Candidate"):
        st.session_state.rows.append(
            {"candidate_id": f"Candidate {len(st.session_state.rows) + 1}", "resume_skills": [], "job_skills": [], "resume_years": 0, "job_years": 0}
        )
with col_b:
    if st.button("🗑️ Reset Batch"):
        st.session_state.rows = [
            {"candidate_id": "Candidate 1", "resume_skills": [], "job_skills": [], "resume_years": 0, "job_years": 0}
        ]

st.write("") # Spacer

for idx, row in enumerate(st.session_state.rows):
    with st.expander(f"👤 {row.get('candidate_id', f'Candidate {idx + 1}')}", expanded=(idx == 0)):
        
        # UI UPGRADE: Side-by-Side Columns for cleaner layout
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.markdown("**Candidate Profile**")
            row["candidate_id"] = st.text_input("Candidate Label", value=row.get("candidate_id", f"Candidate {idx + 1}"), key=f"candidate_id_{idx}")
            row["resume_skills"] = st.multiselect("Candidate Skills", options=all_skills, default=row["resume_skills"], key=f"resume_skills_{idx}")
            row["resume_years"] = st.number_input("Years of Experience", min_value=0.0, max_value=100.0, value=float(row["resume_years"]), step=1.0, key=f"resume_years_{idx}")
            
        with c_right:
            st.markdown("**Job Requirements**")
            # Added a spacer to align fields visually
            st.write("") 
            st.write("") 
            row["job_skills"] = st.multiselect("Required Skills", options=all_skills, default=row["job_skills"], key=f"job_skills_{idx}")
            row["job_years"] = st.number_input("Required Years of Exp", min_value=0.0, max_value=100.0, value=float(row["job_years"]), step=1.0, key=f"job_years_{idx}")

st.write("") # Spacer
run = st.button("🚀 Run ML Prediction", type="primary", use_container_width=True)

# -----------------------------------------
# PREDICTION & RESULTS SECTION
# -----------------------------------------
if run:
    payload = {"items": st.session_state.rows}
    
    # UI UPGRADE: Add a spinner while waiting for the API
    with st.spinner("Analyzing candidate batch via XGBoost..."):
        try:
            resp = requests.post(f"{api_url.rstrip('/')}/predict", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()["rows"]
        except Exception as exc:
            st.error(f"Prediction request failed: {exc}")
            st.stop()

    result_df = pd.DataFrame(data)
    
    if not result_df.empty:
        st.success("Batch processing complete!")
        st.divider()
        
        # UI UPGRADE: The "Hero" Spotlight for the top candidate
        top_candidate = result_df.iloc[0]
        
        st.markdown("### 🏆 Top Match Recommendation")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(label="Best Candidate", value=top_candidate["candidate_id"])
        with metric_col2:
            st.metric(label="AI Confidence Score", value=f"{top_candidate['score']}%")
        with metric_col3:
            # Color code the decision
            decision_color = "🟢" if top_candidate['decision'].lower() == "accepted" else "🔴"
            st.metric(label="System Decision", value=f"{decision_color} {top_candidate['decision'].upper()}")

        st.write("")
        st.markdown("### 📊 Detailed Batch Results")
        
        display_cols = [
        "candidate_id",
        "ranking",
        "decision",
        "score",
        ]
        st.subheader("Results")
        
        result_df["score_bar"] = result_df["score"] / 100

        st.dataframe(
            result_df[["candidate_id", "ranking", "decision", "score", "score_bar"]],
            column_config={
                "score_bar": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0,
                    max_value=1,
                ),
            },
            use_container_width=True,
        )

        st.download_button(
            "📥 Download Results as CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )