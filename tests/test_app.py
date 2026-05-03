import pytest
from fastapi.testclient import TestClient

# Import your FastAPI app from your main file
from app.main import app 

client = TestClient(app)

def test_predict_endpoint():
    """Test that the prediction engine successfully processes a candidate batch."""
    payload = {
        "items": [
            {
                "candidate_id": "Test Candidate 1",
                "resume_skills": ["Python", "SQL", "Pandas"],
                "job_skills": ["Python", "SQL", "AWS", "Pandas"],
                "resume_years": 4.0,
                "job_years": 2.0
            }
        ]
    }
    
    response = client.post("/predict", json=payload)
    
    # 1. Check that the server responded with a 200 OK success code
    assert response.status_code == 200
    
    data = response.json()
    
    # 2. Check that it returned our "rows" array
    assert "rows" in data
    assert len(data["rows"]) == 1
    
    # 3. Verify the final API payload contains the correct output keys
    result_row = data["rows"][0]
    assert result_row["candidate_id"] == "Test Candidate 1"
    assert "score" in result_row
    assert "decision" in result_row
    assert "ranking" in result_row  # Replaced experience_gap with ranking!


def test_feedback_endpoint():
    """Test that the feedback loop successfully accepts data."""
    payload = {
        "candidate_id": "Test Candidate 1",
        "match_score": 0.75,
        "experience_gap": 2.0,
        "final_decision": 1
    }
    
    response = client.post("/feedback", json=payload)
    
    # Check that the server accepted the feedback without a 422 Validation Error
    assert response.status_code == 200