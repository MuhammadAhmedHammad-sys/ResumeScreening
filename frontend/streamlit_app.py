from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (BASE_DIR / ".." / "data" / "skillset.json").resolve()
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="ML Prediction App", layout="wide")
st.title("ML Prediction App")
st.write("Enter resume and job details, then rank candidates by model score.")


def load_skills() -> List[str]:
    if not DATA_PATH.exists():
        st.warning(f"Skill file not found: {DATA_PATH}. Please add data/skillset.json.")
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        skills = json.load(f)
    return sorted([str(s).strip() for s in skills if str(s).strip()])


all_skills = load_skills()

with st.sidebar:
    st.header("API")
    api_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL)
    st.caption("Example: http://127.0.0.1:8000")

if "rows" not in st.session_state:
    st.session_state.rows = [
        {"candidate_id": "Candidate 1", "resume_skills": [], "job_skills": [], "resume_years": 0, "job_years": 0}
    ]

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("Add resume row"):
        st.session_state.rows.append(
            {"candidate_id": f"Candidate {len(st.session_state.rows) + 1}", "resume_skills": [], "job_skills": [], "resume_years": 0, "job_years": 0}
        )
with col_b:
    if st.button("Reset rows"):
        st.session_state.rows = [
            {"candidate_id": "Candidate 1", "resume_skills": [], "job_skills": [], "resume_years": 0, "job_years": 0}
        ]

st.subheader("Candidate batch")
for idx, row in enumerate(st.session_state.rows):
    with st.expander(f"{row.get('candidate_id', f'Candidate {idx + 1}')}", expanded=(idx == 0)):
        row["candidate_id"] = st.text_input("Candidate label", value=row.get("candidate_id", f"Candidate {idx + 1}"), key=f"candidate_id_{idx}")
        row["resume_skills"] = st.multiselect(
            "Resume skills",
            options=all_skills,
            default=row["resume_skills"],
            key=f"resume_skills_{idx}",
        )
        row["job_skills"] = st.multiselect(
            "Job skills",
            options=all_skills,
            default=row["job_skills"],
            key=f"job_skills_{idx}",
        )
        c1, c2 = st.columns(2)
        with c1:
            row["resume_years"] = st.number_input(
                "Resume years of experience",
                min_value=0.0,
                max_value=100.0,
                value=float(row["resume_years"]),
                step=1.0,
                key=f"resume_years_{idx}",
            )
        with c2:
            row["job_years"] = st.number_input(
                "Required years of experience",
                min_value=0.0,
                max_value=100.0,
                value=float(row["job_years"]),
                step=1.0,
                key=f"job_years_{idx}",
            )

run = st.button("Predict", type="primary")

if run:
    payload = {"items": st.session_state.rows}
    try:
        resp = requests.post(f"{api_url.rstrip('/')}/predict", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()["rows"]
    except Exception as exc:
        st.error(f"Prediction request failed: {exc}")
        st.stop()

    result_df = pd.DataFrame(data)
    if not result_df.empty:
        display_cols = [
            "candidate_id",
            "ranking",
            "decision",
            "score",
            "probability_accepted",
            "probability_rejected",
            "matchscore",
            "age_gap",
        ]
        st.subheader("Results")
        st.dataframe(result_df[display_cols], use_container_width=True)
        st.download_button(
            "Download results as CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
