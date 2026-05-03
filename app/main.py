from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .prediction import predict_batch

app = FastAPI(title="Resume Match Prediction API", version="1.0.0")

# -----------------------------------------
# DATA MODELS
# -----------------------------------------
class ResumeItem(BaseModel):
    candidate_id: str | None = None
    resume_skills: List[str] = Field(default_factory=list)
    job_skills: List[str] = Field(default_factory=list)
    resume_years: float
    job_years: float

class BatchRequest(BaseModel):
    items: List[ResumeItem]

class PredictionRow(BaseModel):
    candidate_id: str | None = None
    ranking: int
    score: float
    decision: str

class BatchResponse(BaseModel):
    rows: List[PredictionRow]

class FeedbackItem(BaseModel):
    candidate_id: str
    match_score: float
    experience_gap: float
    final_decision: int  

# -----------------------------------------
# API ENDPOINTS
# -----------------------------------------
@app.get("/health")
def health():
    """
    Checks if the FastAPI server is running successfully. 
    """
    return {"status": "ok"}


@app.post("/predict", response_model=BatchResponse)
def predict(request: BatchRequest):
    """
    Processes candidate batch through XGBoost and returns rankings and decisions.
    """
    if not request.items:
        raise HTTPException(status_code=400, detail="No records provided")

    df = predict_batch(
        candidate_ids=[item.candidate_id for item in request.items],
        resume_skills=[item.resume_skills for item in request.items],
        job_skills=[item.job_skills for item in request.items],
        resume_years=[item.resume_years for item in request.items],
        job_years=[item.job_years for item in request.items],
    )
    
    rows = df[["candidate_id", "ranking", "decision", "score"]].to_dict(orient="records")
    return {"rows": rows}


@app.post("/feedback")
def submit_feedback(feedback: FeedbackItem):
    """
    Saves the human recruiter's final hiring decision to feedback.csv
    using only the numeric features expected by the training pipeline.
    """
    base_dir = Path(__file__).resolve().parent
    feedback_path = (base_dir / ".." / "data" / "feedback.csv").resolve()
    
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    
    new_data = pd.DataFrame([{
        "match_score": feedback.match_score,
        "experience_gap": feedback.experience_gap,
        "label": feedback.final_decision
    }])
    
    if not feedback_path.exists():
        new_data.to_csv(feedback_path, index=False)
    else:
        new_data.to_csv(feedback_path, mode='a', header=False, index=False)
        
    return {"status": "success", "message": f"Feedback for {feedback.candidate_id} saved."}