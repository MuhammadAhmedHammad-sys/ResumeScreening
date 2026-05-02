import pytest
from fastapi.testclient import TestClient

# Import your FastAPI app from your main file
# (Assuming your app instance is named 'app' inside app/main.py)
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
    
    # 3. Verify the math and features are correct based on our recent updates!
    result_row = data["rows"][0]
    assert result_row["candidate_id"] == "Test Candidate 1"
    assert "score" in result_row
    assert "decision" in result_row
    
    # Verify our specific feature renaming was successful
    assert "experience_gap" in result_row
    assert result_row["experience_gap"] == 2.0  # 4.0 resume_years - 2.0 job_years


def test_feedback_endpoint():
    """Test that the feedback loop successfully accepts data."""
    # Notice: We removed parsed_resume here, just like we did in the frontend!
    payload = {
        "candidate_id": "Test Candidate 1",
        "match_score": 0.75,
        "experience_gap": 2.0,
        "final_decision": 1
    }
    
    response = client.post("/feedback", json=payload)
    
    # Check that the server accepted the feedback without a 422 Validation Error
    assert response.status_code == 200