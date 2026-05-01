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
    candidate_ids=[item.candidate_id for item in request.items],
    resume_skills=[item.resume_skills for item in request.items],
    job_skills=[item.job_skills for item in request.items],
    resume_years=[item.resume_years for item in request.items],
    job_years=[item.job_years for item in request.items],
)
    
    rows = df[["candidate_id", "ranking", "decision", "score"]].to_dict(orient="records")
    return {"rows": rows}
