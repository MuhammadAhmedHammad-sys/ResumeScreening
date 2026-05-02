import pandas as pd
import numpy as np
import os
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from prefect import task, flow
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PREFECT TASKS
# ---------------------------------------------------------

@task(name="Load Dataset", retries=2, retry_delay_seconds=10)
def load_data(file_path: str):
    """Loads the raw CSV dataset."""
    print(f"Loading data from: {file_path}")
    return pd.read_csv(file_path)

@task(name="Parse Resumes")
def parse_row(text: str):
    """Extracts skills and experience from the raw text block."""
    resume_part, job_part = text.split("[sep] job:")
    resume_part = resume_part.replace("resume:", "").strip()
    job_part = job_part.strip()
    job_role, job_desc = job_part.split(":", 1)
    job_role = job_role.strip()
    ignore_phrases = ["with focus", "for", "in a", "requires", "experience"]

    job_skills = [
        x.strip().lower() for x in job_desc.split(",") 
        if not any(p in x.lower() for p in ignore_phrases)
    ]

    job_experience = 0
    for x in job_skills:
        if "year" in x or "senior" in x:
            num = ''.join(filter(str.isdigit, x))
            if num:
                job_experience = int(num)
            elif "senior" in x:
                job_experience = 7

    tokens = [x.strip() for x in resume_part.split(",")]
    resume_skills = []
    experience = 0
    noise_prefixes = ["experienced in", "expert in", "versed in", "contributed to", "accomplished in", "proficient in", "specialized in", "trained in", "skilled in"]

    for t in tokens:
        t_lower = t.lower()
        if "year" in t_lower or "senior" in t_lower:
            num = ''.join(filter(str.isdigit, t_lower))
            if num:
                experience = int(num)
            elif "senior" in t_lower:
                experience = 7
        else:
            for p in noise_prefixes:
                t_lower = t_lower.replace(p, "").strip()
            resume_skills.append(t_lower)

    job_skills = [x for x in job_skills if "year" not in x and "senior" not in x]
    resume_skills = [x for x in resume_skills if "year" not in x and "senior" not in x]

    return {
        "resume_skills": resume_skills,
        "experience_years": experience,
        "job_skills": job_skills,
        "job_experience_years": job_experience
    }

@task(name="Calculate Feature Engineering")
def engineer_features(df: pd.DataFrame):
    """Calculates the Match Score and Experience Gap features."""
    def skill_match_score(resume_skills, job_skills):
        resume_set = set(resume_skills)
        job_set = set(job_skills)
        if len(job_set) == 0: return 0.0
        return len(resume_set & job_set) / len(job_set)

    df_parsed = df['text'].apply(parse_row).apply(pd.Series)
    df_tabular = pd.concat([df_parsed, df['label']], axis=1)

    df_tabular['match_score'] = df_tabular.apply(lambda row: skill_match_score(row['resume_skills'], row['job_skills']), axis=1)
    df_tabular['experience_gap'] = df_tabular['experience_years'] - df_tabular['job_experience_years']
    
    return df_tabular[['match_score', 'experience_gap', 'label']]

# --- NEW TASK: MERGE FEEDBACK ---
@task(name="Merge Human Feedback")
def merge_feedback(df_raw_features: pd.DataFrame, feedback_path: str):
    """Checks if feedback.csv exists, and if so, merges it with the historical data."""
    if os.path.exists(feedback_path):
        print(f"Found human feedback at {feedback_path}. Merging data...")
        df_feedback = pd.read_csv(feedback_path)
        
        # Combine the old data with the new human-labeled data
        combined_df = pd.concat([df_raw_features, df_feedback], ignore_index=True)
        print(f"Added {len(df_feedback)} new manual feedback records to training set.")
        return combined_df
    else:
        print("No feedback.csv found. Training strictly on raw historical data.")
        return df_raw_features
# --------------------------------

@task(name="Train XGBoost Model")
def train_model(df_features: pd.DataFrame):
    """Trains the XGBoost model and evaluates its accuracy."""
    X = df_features.drop(columns='label')
    y = df_features['label']
    
    # 1. Split data (80% for learning, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    print("Training XGBoost...")
    model.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # NEW: EVALUATE THE MODEL
    # ---------------------------------------------------------
    print("\n--- 📊 MODEL PERFORMANCE REPORT ---")
    
    # Ask the model to predict the answers for the 20% test data it hasn't seen
    y_pred = model.predict(X_test)
    
    # Calculate how well it did
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    print(f"Accuracy:  {accuracy * 100:.2f}% (Overall correctness)")
    print(f"Precision: {precision * 100:.2f}% (When it says HIRE, how often is it right?)")
    print(f"Recall:    {recall * 100:.2f}% (Out of all good candidates, how many did it find?)")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=["Reject", "Hire"], zero_division=0))
    print("-----------------------------------\n")
    
    return model

@task(name="Save Artifacts")
def save_model(model, output_path: str):
    """Saves the trained model to the models directory."""
    print(f"Saving model to {output_path}...")
    joblib.dump(model, output_path)
    print("Pipeline Complete! ✅")

# ---------------------------------------------------------
# PREFECT FLOW (The Conductor)
# ---------------------------------------------------------

@flow(name="Resume Screening - Retraining Pipeline", log_prints=True)
def run_training_pipeline():
    """This is the main conductor function that runs all the tasks in order."""
    
    # Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw.csv")
    feedback_path = os.path.join(base_dir, "data", "feedback.csv")  # <-- Added feedback path
    model_output_path = os.path.join(base_dir, "models", "xgb_v2.pkl")

    # Step 1: Load Data
    raw_df = load_data(data_path)

    # Step 2: Parse text and engineer features from historical data
    df_features = engineer_features(raw_df)

    # Step 3: Merge in the new recruiter feedback! (The Continuous Learning step)
    final_dataset = merge_feedback(df_features, feedback_path)

    # Step 4: Train the model on the combined data
    trained_model = train_model(final_dataset)

    # Step 5: Save the output (overwrites xgb_v2.pkl with the smarter model)
    save_model(trained_model, model_output_path)

# Run the flow if the script is executed directly
if __name__ == "__main__":
    run_training_pipeline()