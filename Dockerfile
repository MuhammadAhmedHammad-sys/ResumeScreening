# 1. Start with a lightweight, official Python image (This works on Mac/Win/Linux!)
FROM python:3.10-slim

# 2. Stop Python from generating .pyc files and enable real-time terminal output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Create a working directory inside the container
WORKDIR /app

# 4. Install system tools that ML libraries (like XGBoost) sometimes need to compile
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy your requirements file and install dependencies
# (Make sure you have a requirements.txt file with fastapi, streamlit, xgboost, etc.)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your entire project into the container
COPY . .

# 7. Expose the ports your apps will run on
EXPOSE 8000 8501
