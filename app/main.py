from __future__ import annotations

from typing import List, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .prediction import predict_batch

app = FastAPI(title="Resume Match Prediction API", version="1.0.0")


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
    matchscore: float
    age_gap: float
    prediction: int | str
    probability_accepted: float
    probability_rejected: float
    score: float
    decision: str


class BatchResponse(BaseModel):
    rows: List[PredictionRow]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=BatchResponse)
def predict(request: BatchRequest):
    if not request.items:
        raise HTTPException(status_code=400, detail="No records provided")

    df = predict_batch(
        resume_skills=[item.resume_skills for item in request.items],
        job_skills=[item.job_skills for item in request.items],
        resume_years=[item.resume_years for item in request.items],
        job_years=[item.job_years for item in request.items],
    )

    rows = df.to_dict(orient="records")
    for idx, row in enumerate(rows):
        row["candidate_id"] = request.items[idx].candidate_id or f"Candidate {idx + 1}"
    return {"rows": rows}
