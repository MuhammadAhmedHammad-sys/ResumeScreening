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

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

def load_skills() -> List[str]:
    if not DATA_PATH.exists():
        st.warning(f"Skill file not found: {DATA_PATH}. Please add data/skillset.json.")
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        skills = json.load(f)
    return sorted([str(s).strip() for s in skills if str(s).strip()])

all_skills = load_skills()

# --- MATH FUNCTION ---
def calculate_match_score(resume_skills: List[str], job_skills: List[str]) -> float:
    """Calculates Jaccard Similarity between two lists of skills."""
    if not job_skills and not resume_skills:
        return 0.0
    
    set_resume = set(s.lower() for s in resume_skills)
    set_job = set(s.lower() for s in job_skills)
    
    intersection = len(set_resume.intersection(set_job))
    union = len(set_resume.union(set_job))
    
    return round(intersection / union, 2) if union > 0 else 0.0

# --- NEW: CALLBACK FUNCTION FOR BUTTONS ---
def handle_feedback(cid: str, match_score: float, exp_gap: float, decision: int, api_endpoint: str):
    """Sends data to backend and updates Streamlit memory synchronously before the UI re-renders."""
    payload = {
        "candidate_id": cid,
        "match_score": float(match_score),
        "experience_gap": float(exp_gap),
        "final_decision": int(decision)
    }
    try:
        requests.post(f"{api_endpoint.rstrip('/')}/feedback", json=payload, timeout=5)
    except Exception as e:
        print(f"Error saving feedback: {e}")
        
    # Add to memory so the button hides immediately on the next render
    st.session_state.submitted_feedback.add(cid)

# -----------------------------------------
# SIDEBAR
# -----------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/942/942748.png", width=80)
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

st.write("") 

for idx, row in enumerate(st.session_state.rows):
    with st.expander(f"👤 {row.get('candidate_id', f'Candidate {idx + 1}')}", expanded=(idx == 0)):
        
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.markdown("**Candidate Profile**")
            row["candidate_id"] = st.text_input("Candidate Label", value=row.get("candidate_id", f"Candidate {idx + 1}"), key=f"candidate_id_{idx}")
            row["resume_skills"] = st.multiselect("Candidate Skills", options=all_skills, default=row["resume_skills"], key=f"resume_skills_{idx}")
            row["resume_years"] = st.number_input("Years of Experience", min_value=0.0, max_value=100.0, value=float(row["resume_years"]), step=1.0, key=f"resume_years_{idx}")
            
        with c_right:
            st.markdown("**Job Requirements**")
            st.write("") 
            st.write("") 
            row["job_skills"] = st.multiselect("Required Skills", options=all_skills, default=row["job_skills"], key=f"job_skills_{idx}")
            row["job_years"] = st.number_input("Required Years of Exp", min_value=0.0, max_value=100.0, value=float(row["job_years"]), step=1.0, key=f"job_years_{idx}")

st.write("")
run = st.button("🚀 Run ML Prediction", type="primary", use_container_width=True)

# -----------------------------------------
# PREDICTION & RESULTS SECTION
# -----------------------------------------
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
    
if "submitted_feedback" not in st.session_state:
    st.session_state.submitted_feedback = set()

if run:
    payload = {"items": st.session_state.rows}
    
    with st.spinner("Analyzing candidate batch via XGBoost..."):
        try:
            resp = requests.post(f"{api_url.rstrip('/')}/predict", json=payload, timeout=60)
            resp.raise_for_status()
            
            st.session_state.prediction_data = resp.json()["rows"]
            st.session_state.submitted_feedback.clear() 
            
        except Exception as exc:
            st.error(f"Prediction request failed: {exc}")
            st.stop()

if st.session_state.prediction_data is not None:
    result_df = pd.DataFrame(st.session_state.prediction_data)
    
    if not result_df.empty:
        st.success("Batch processing complete!")
        st.divider()
        
        top_candidate = result_df.iloc[0]
        
        st.markdown("### 🏆 Top Match Recommendation")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(label="Best Candidate", value=top_candidate["candidate_id"])
        with metric_col2:
            top_score_display = round(float(top_candidate['score']), 2)
            st.metric(label="AI Confidence Score", value=f"{top_score_display}%")
        with metric_col3:
            decision_color = "🟢" if top_candidate['decision'].lower() == "accepted" else "🔴"
            st.metric(label="System Decision", value=f"{decision_color} {top_candidate['decision'].upper()}")

        st.write("")
        st.divider()
        
        # -----------------------------------------
        # ALL CANDIDATES FEEDBACK LOOP
        # -----------------------------------------
        st.markdown("### 👨‍💼 Batch Review & Ground Truth")
        st.info("Review each candidate. Buttons will disappear immediately after you click them to prevent duplicate entries.")
        
        for idx, row in result_df.iterrows():
            cid = str(row["candidate_id"])
            
            # Get user input for this candidate
            original_candidate = next((item for item in st.session_state.rows if str(item["candidate_id"]) == cid), None)
            
            # Calculate real data
            if original_candidate:
                real_match_score = calculate_match_score(original_candidate["resume_skills"], original_candidate["job_skills"])
                real_exp_gap = float(original_candidate["resume_years"]) - float(original_candidate["job_years"])
            else:
                real_match_score = 0.0
                real_exp_gap = 0.0
                
            score_display = round(float(row['score']), 2)
            st.markdown(f"**👤 {cid}** | AI Score: **{score_display}%** | Actual Match: **{real_match_score}** | Exp Gap: **{real_exp_gap}**")
            
            # Hide buttons if already clicked
            if cid in st.session_state.submitted_feedback:
                st.success(f"✅ Ground truth saved for {cid}")
            else:
                fb_col1, fb_col2, _ = st.columns([1, 1, 3])
                
                with fb_col1:
                    # Professional Streamlit Callback method
                    st.button(
                        f"Hire {cid}", 
                        key=f"hire_{cid}", 
                        type="primary",
                        on_click=handle_feedback,
                        args=(cid, real_match_score, real_exp_gap, 1, api_url)
                    )
                        
                with fb_col2:
                    st.button(
                        f"Reject {cid}", 
                        key=f"reject_{cid}",
                        on_click=handle_feedback,
                        args=(cid, real_match_score, real_exp_gap, 0, api_url)
                    )
            st.write("---")

        # -----------------------------------------
        st.markdown("### 📊 Detailed Batch Results")
        available_cols = [c for c in ["candidate_id", "ranking", "decision", "score", "probability_accepted", "probability_rejected", "age_gap"] if c in result_df.columns]
        st.dataframe(result_df[available_cols], use_container_width=True, hide_index=True)
        
        st.download_button(
            "📥 Download Results as CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )