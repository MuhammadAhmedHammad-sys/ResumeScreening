from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / ".." / "models" / "xgb.pkl").resolve()


def load_model():
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Place the trained xgb.pkl file there."
        )
    return joblib.load(MODEL_PATH)


def normalize_skills(skills: Iterable[str] | str | None) -> List[str]:
    """Normalize a skill input into a clean list of unique strings."""
    if skills is None:
        return []

    if isinstance(skills, str):
        raw = [s.strip() for s in skills.split(",")]
    else:
        raw = [str(s).strip() for s in skills]

    cleaned = [s for s in raw if s]
    # Preserve order while removing duplicates.
    seen = set()
    unique = []
    for skill in cleaned:
        key = skill.lower()
        if key not in seen:
            seen.add(key)
            unique.append(skill)
    return unique


def match_score(resume_skills: Iterable[str] | str | None, job_skills: Iterable[str] | str | None) -> float:
    """Fraction of required job skills matched by the resume."""
    resume_set = {s.lower() for s in normalize_skills(resume_skills)}
    job_set = {s.lower() for s in normalize_skills(job_skills)}

    if not job_set:
        return 0.0

    return len(resume_set & job_set) / len(job_set)


def age_gap(app_years: float, req_years: float) -> float:
    """Numeric gap between applicant years and required years."""
    return float(app_years) - float(req_years)


def build_feature_frame(
    resume_skills: Sequence[Iterable[str] | str],
    job_skills: Sequence[Iterable[str] | str],
    resume_years: Sequence[float],
    job_years: Sequence[float],
) -> pd.DataFrame:
    """Create the model input dataframe for a batch of applicants."""
    rows = []
    for rs, js, ry, jy in zip(resume_skills, job_skills, resume_years, job_years):
        rows.append(
            {
                "matchscore": match_score(rs, js),
                "age_gap": age_gap(ry, jy),
            }
        )
    return pd.DataFrame(rows, columns=["matchscore", "age_gap"])


def _positive_class_index(model) -> int:
    """Pick the positive-class column from predict_proba."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        return 1

    classes = list(classes)
    for preferred in (1, "1", True, "accepted", "accept"):
        if preferred in classes:
            return classes.index(preferred)

    return len(classes) - 1


def predict_batch(
    candidate_ids: Sequence[str | None],
    resume_skills: Sequence[Iterable[str] | str],
    job_skills: Sequence[Iterable[str] | str],
    resume_years: Sequence[float],
    job_years: Sequence[float],
) -> pd.DataFrame:
    clf = load_model()
    X = build_feature_frame(resume_skills, job_skills, resume_years, job_years)

    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)
    pos_idx = _positive_class_index(clf)

    result_df = X.copy()
    result_df["candidate_id"] = [
        cid if cid else f"Candidate {i+1}" for i, cid in enumerate(candidate_ids)
    ]

    result_df["prediction"] = predictions
    result_df["probability_accepted"] = probabilities[:, pos_idx]
    result_df["probability_rejected"] = 1 - result_df["probability_accepted"]
    result_df["score"] = (result_df["probability_accepted"] * 100).round(2)
    result_df["ranking"] = result_df["score"].rank(method="dense", ascending=False).astype(int)
    result_df["decision"] = result_df["prediction"].apply(
        lambda x: "accepted" if str(x) in {"1", "True", "true", "accepted", "accept"} or x == 1 else "rejected"
    )

    return result_df.sort_values(["ranking", "score"], ascending=[True, False]).reset_index(drop=True)