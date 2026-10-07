import streamlit as st
import joblib
import pandas as pd

# -------------------------------------------------------------
# Load the trained model and reference medians (saved from the notebook)
# -------------------------------------------------------------
model = joblib.load("rf_model.pkl")
feature_medians = joblib.load("feature_medians.pkl")

st.set_page_config(page_title="CGPA Predictor", page_icon="🎓")
st.title("🎓 CGPA Prediction & Academic Optimization Tool")
st.write(
    "Enter your academic and behavioural details below. The tool will predict your "
    "current CGPA and suggest which areas may be worth focusing on."
)

st.caption(
    "This tool does not store or share your answers — predictions happen only in your "
    "current browser session."
)

st.subheader("Academic history")
previous_cgpa = st.number_input("Previous CGPA (before your last completed semester)", 0.0, 5.0, 3.0, 0.01)
units_taken = st.number_input("Total units taken", 10, 24, 18)
ca_score = st.number_input("Average CA score (out of 40)", 0.0, 40.0, 25.0)
higher_units = st.number_input("Number of higher-unit (3-unit or more) courses", 0, 10, 3)

st.subheader("Attitude & mindset")
test_prep = st.slider("Test preparedness (1 = not prepared, 5 = very prepared)", 1, 5, 3)
assignment_commitment = st.slider("Assignment commitment (1 = low, 5 = high)", 1, 5, 3)
procrastination = st.slider("Procrastination tendency (1 = never, 5 = always)", 1, 5, 3)

st.subheader("Study habits & behaviour")
study_hours = st.number_input("Weekly study hours", 0.0, 112.0, 10.0)
attendance = st.slider("Attendance rate (1 = low, 5 = high)", 1, 5, 3)
extracurricular = st.number_input("Extracurricular hours per week", 0.0, 40.0, 2.0)

st.subheader("Resource access")
access_pq = st.selectbox("Do you have access to past questions / course materials?", ["Yes", "No"])
access_pq_val = 1.0 if access_pq == "Yes" else 0.0

input_df = pd.DataFrame([{
    "Previous CGPA": previous_cgpa,
    "Units taken": units_taken,
    "Average CA score": ca_score,
    "No of higher unit course": higher_units,
    "Test preparedness": test_prep,
    "Assignment commitment": assignment_commitment,
    "Procrastination tendency": procrastination,
    "Weekly Study hours": study_hours,
    "Atendance rate": attendance,
    "Extracurricular hours": extracurricular,
    "Access to PQ": access_pq_val,
}])

if st.button("Predict my CGPA"):
    prediction = model.predict(input_df)[0]
    st.success(f"**Predicted CGPA: {prediction:.2f}**")

    tips = []
    if attendance < feature_medians["Atendance rate"]:
        tips.append(
            "Your attendance is below the typical level in this dataset — attendance had the "
            "strongest link to CGPA among behavioural factors, so improving it could meaningfully help."
        )
    if procrastination > feature_medians["Procrastination tendency"]:
        tips.append(
            "Your procrastination rating is higher than typical — lower procrastination was "
            "linked to better outcomes."
        )
    if assignment_commitment < feature_medians["Assignment commitment"]:
        tips.append(
            "Your assignment commitment is below typical — staying consistent with assignments "
            "was linked to better CGPA."
        )
    if ca_score < feature_medians["Average CA score"]:
        tips.append(
            "Your CA score is below typical — since CA feeds directly into your overall result, "
            "focusing here could help."
        )

    st.subheader("Suggestions")
    if tips:
        for t in tips:
            st.write("•", t)
    else:
        st.write("Your behavioural indicators are already at or above typical levels in this dataset — keep it up!")

    st.caption(
        "Note: this prediction is based on a small sample (~37 responses) and should be read "
        "as a rough guide, not a guarantee."
    )
