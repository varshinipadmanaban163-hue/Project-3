
import streamlit as st
import pandas as pd
import joblib

st.title("Job Placement Prediction")

model = joblib.load("log_model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("ohe.pkl")
feature_columns = joblib.load("feature_columns.pkl")

with st.form("placement_form"):

    c1, c2, c3 = st.columns(3)

    with c1:
        age_years = st.number_input("Age", 18, 65, 25)
        gender = st.selectbox("Gender", ["Female", "Male"])
        ssc_percentage = st.number_input("SSC Percentage", 0.0, 100.0, 75.0)
        hsc_percentage = st.number_input("HSC Percentage", 0.0, 100.0, 75.0)
        degree_percentage = st.number_input("Degree Percentage", 0.0, 100.0, 75.0)
        degree_specialization = st.selectbox(
            "Degree Specialization", list(ohe.categories_[0])
        )
        company_tier = st.selectbox(
            "Company Tier", list(ohe.categories_[1])
        )
        competition_level = st.selectbox(
            "Competition Level", list(ohe.categories_[2])
        )

    with c2:
        technical_score = st.number_input("Technical Score", 0.0, 100.0, 70.0)
        aptitude_score = st.number_input("Aptitude Score", 0.0, 100.0, 70.0)
        communication_score = st.number_input(
            "Communication Score", 0.0, 100.0, 70.0
        )
        skills_match_percentage = st.number_input(
            "Skills Match Percentage", 0.0, 100.0, 70.0
        )
        certifications_count = st.number_input(
            "Certifications Count", 0, 50, 2
        )
        internship_experience = st.selectbox(
            "Internship Experience", ["No", "Yes"]
        )
        years_of_experience = st.number_input(
            "Years of Experience", 0, 40, 2
        )
        career_switch_willingness = st.selectbox(
            "Career Switch Willingness", ["No", "Yes"]
        )

    with c3:
        relevant_experience = st.selectbox(
            "Relevant Experience", ["No", "Yes"]
        )
        previous_ctc_lpa = st.number_input("Previous CTC (LPA)", 0.0, 200.0, 3.0)
        expected_ctc_lpa = st.number_input("Expected CTC (LPA)", 0.0, 200.0, 5.0)
        job_role_match = st.selectbox(
            "Job Role Match", ["Matched", "Not Matched"]
        )
        bond_requirement = st.selectbox(
            "Bond Requirement", ["Not Required", "Required"]
        )
        notice_period_days = st.number_input(
            "Notice Period (Days)", 0, 180, 30
        )
        layoff_history = st.selectbox("Layoff History", ["No", "Yes"])
        employment_gap_months = st.number_input(
            "Employment Gap (Months)", 0, 120, 0
        )
        relocation_willingness = st.selectbox(
            "Relocation Willingness", ["No", "Yes"]
        )

    submit = st.form_submit_button("Predict Placement")


if submit:

    # Create candidate data
    data = pd.DataFrame([{
        "age_years": age_years,
        "gender": gender,
        "ssc_percentage": ssc_percentage,
        "hsc_percentage": hsc_percentage,
        "degree_percentage": degree_percentage,
        "degree_specialization": degree_specialization,
        "technical_score": technical_score,
        "aptitude_score": aptitude_score,
        "communication_score": communication_score,
        "skills_match_percentage": skills_match_percentage,
        "certifications_count": certifications_count,
        "internship_experience": internship_experience,
        "years_of_experience": years_of_experience,
        "career_switch_willingness": career_switch_willingness,
        "relevant_experience": relevant_experience,
        "previous_ctc_lpa": previous_ctc_lpa,
        "expected_ctc_lpa": expected_ctc_lpa,
        "company_tier": company_tier,
        "job_role_match": job_role_match,
        "competition_level": competition_level,
        "bond_requirement": bond_requirement,
        "notice_period_days": notice_period_days,
        "layoff_history": layoff_history,
        "employment_gap_months": employment_gap_months,
        "relocation_willingness": relocation_willingness
    }])

    # Encode categorical columns using the same alphabetical
    # ordering as LabelEncoder
    label_columns = [
        "gender", "internship_experience",
        "career_switch_willingness", "relevant_experience",
        "job_role_match", "bond_requirement",
        "layoff_history", "relocation_willingness"
    ]

    for col in label_columns:
        categories = sorted(data[col].dropna().unique())
        # Use the complete category list for binary columns
        if col == "gender":
            categories = ["Female", "Male"]
        elif col == "job_role_match":
            categories = ["Matched", "Not Matched"]
        elif col == "bond_requirement":
            categories = ["Not Required", "Required"]
        else:
            categories = ["No", "Yes"]

        data[col] = data[col].map(
            {value: index for index, value in enumerate(categories)}
        )

    # Create the same derived columns used during training
    data["interview_score"] = data[
        ["technical_score", "aptitude_score", "communication_score"]
    ].mean(axis=1)

    data["experience_category"] = 0
    data.loc[data["years_of_experience"].between(1, 3),
             "experience_category"] = 1
    data.loc[data["years_of_experience"] >= 4,
             "experience_category"] = 2

    data["academic_average"] = data[
        ["ssc_percentage", "hsc_percentage", "degree_percentage"]
    ].mean(axis=1)

    data["academic_performance"] = 0
    data.loc[data["academic_average"].between(60, 80),
             "academic_performance"] = 1
    data.loc[data["academic_average"] >= 80,
             "academic_performance"] = 2

    data["interview_performance"] = 0
    data.loc[data["interview_score"].between(60, 80),
             "interview_performance"] = 1
    data.loc[data["interview_score"] >= 80,
             "interview_performance"] = 2

    # One-hot encode using the fitted encoder
    onehot_columns = [
        "degree_specialization", "company_tier", "competition_level"
    ]

    encoded = ohe.transform(data[onehot_columns])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(onehot_columns),
        index=data.index
    )

    data = pd.concat(
        [data.drop(columns=onehot_columns), encoded_df], axis=1
    )

    data = data.reindex(columns=feature_columns, fill_value=0)


    data_scaled = scaler.transform(data)


    prediction = model.predict(data_scaled)[0]

    if prediction == 1:
        st.success("Prediction: Placed")
    else:
        st.error("Prediction: Not Placed")
