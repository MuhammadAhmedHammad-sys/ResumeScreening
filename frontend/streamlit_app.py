import streamlit as st
import matplotlib.pyplot as plt

# Function to calculate the match score
def match_score(
        resume_skills,
        job_skills
        ):
    
    resume_set = set(resume_skills)
    job_set = set(job_skills)
    
    if len(job_set) == 0:
        return 0.0
    
    intersection = resume_set & job_set
    score = len(intersection) / len(job_set)
    
    return score

# Streamlit UI setup
st.title("ML Prediction App")
st.write("Enter feature values and get a prediction from the model.")

# Inputs for the sets (skills) and numeric features
skills = ["Python", "Java", "SQL", "Machine Learning", "Data Science", "Deep Learning", "NLP"]
resume_skills = st.multiselect("Select resume skills",skills )
job_skills = st.multiselect("Select job skills", skills)

applicant_experience = st.slider("Years of experience", min_value=0, max_value=32, step=1, key='app_exp')
required_experience = st.slider("Years of experience", min_value=0, max_value=32, step=1, key='req_exp')


# Calculate match score
score = match_score(resume_skills, job_skills)

# Display the match score
st.write(f"Match score: {score * 100:.2f}%")

# Plotting the donut chart to represent match score
fig, ax = plt.subplots(figsize=(3, 3))

# Donut chart
ax.pie([score, 1 - score], labels=["Match", "No Match"], autopct='%1.1f%%', startangle=90, colors=["#2c4fa0", "#A6A6A6"], wedgeprops={'width': 0.4})

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')  

# Add title to the chart
ax.set_title("Match Score as Donut Chart")

# Show the plot
st.pyplot(fig)

# Optionally: Show a brief interpretation of the result
if score > 0.8:
    st.write("The match is very good!")
elif score > 0.5:
    st.write("The match is average.")
else:
    st.write("The match is low.")