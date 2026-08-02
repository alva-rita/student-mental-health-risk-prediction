from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Mental Health Risk Prediction",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# FILE PATHS
# ============================================================

# app.py should be located in the main project folder.
# The trained model files should be inside the models folder.

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# LOAD TRAINED MODEL FILES
# ============================================================

@st.cache_resource
def load_model_files():
    depression_model = joblib.load(
        MODELS_DIR / "depression_logistic_regression.pkl"
    )

    depression_scaler = joblib.load(
        MODELS_DIR / "depression_scaler.pkl"
    )

    depression_feature_names = joblib.load(
        MODELS_DIR / "depression_feature_names.pkl"
    )

    stress_model = joblib.load(
        MODELS_DIR / "stress_logistic_regression.pkl"
    )

    stress_scaler = joblib.load(
        MODELS_DIR / "stress_scaler.pkl"
    )

    stress_feature_names = joblib.load(
        MODELS_DIR / "stress_feature_names.pkl"
    )

    return (
        depression_model,
        depression_scaler,
        depression_feature_names,
        stress_model,
        stress_scaler,
        stress_feature_names,
    )


try:
    (
        depression_model,
        depression_scaler,
        depression_feature_names,
        stress_model,
        stress_scaler,
        stress_feature_names,
    ) = load_model_files()

except FileNotFoundError as error:
    st.error(
        "One or more model files could not be found. "
        "Confirm that the six .pkl files are inside the models folder."
    )
    st.code(str(error))
    st.stop()

except Exception as error:
    st.error("The model files could not be loaded.")
    st.code(str(error))
    st.stop()


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🧠 Student Mental Health Risk Prediction")

st.write(
    """
    This application uses trained machine-learning models to estimate
    depression risk and stress level from information provided by the user.
    """
)

st.info(
    """
    This tool was developed as part of an academic machine-learning project.
    Its output is not a medical diagnosis and should not replace assessment
    or advice from a qualified healthcare professional.
    """
)

assessment_type = st.radio(
    "Select an assessment",
    options=[
        "Depression Risk Assessment",
        "Stress Level Assessment",
    ],
    horizontal=True,
)

st.divider()


# ============================================================
# DEPRESSION ASSESSMENT
# ============================================================

if assessment_type == "Depression Risk Assessment":

    st.header("Depression Risk Assessment")

    st.caption(
        "Complete all fields and select Predict Depression Risk."
    )

    # --------------------------------------------------------
    # CATEGORY ENCODING MAPS
    # --------------------------------------------------------

    gender_map = {
        "Female": 0,
        "Male": 1,
    }

    city_map = {
        "North India": 1,
        "South India": 2,
        "West India": 3,
        "East India": 4,
        "Central India": 5,
        "Others": 6,
    }

    profession_map = {
        "Student": 1,
        "Healthcare": 2,
        "Engineering": 3,
        "Service": 4,
        "Creative": 5,
        "Legal": 6,
        "Education": 7,
        "Business": 8,
        "Tech": 9,
    }

    sleep_duration_map = {
        "Less than 5 hours": 0,
        "5–6 hours": 1,
        "7–8 hours": 2,
        "8–9 hours": 3,
        "More than 9 hours": 4,
    }

    dietary_habits_map = {
        "Healthy": 1,
        "Moderate": 2,
        "Unhealthy": 3,
    }

    degree_field_map = {
        "Engineering": 1,
        "Medical": 2,
        "Business": 3,
        "Science": 4,
        "Arts": 5,
        "Law": 6,
        "Computer": 7,
        "Others": 8,
    }

    yes_no_map = {
        "No": 0,
        "Yes": 1,
    }

    # --------------------------------------------------------
    # DEPRESSION INPUT FORM
    # --------------------------------------------------------

    with st.form("depression_assessment_form"):

        column_one, column_two, column_three = st.columns(3)

        with column_one:

            gender = st.selectbox(
                "Gender",
                options=list(gender_map.keys()),
            )

            age = st.number_input(
                "Age",
                min_value=15,
                max_value=100,
                value=22,
                step=1,
            )

            city = st.selectbox(
                "Location",
                options=list(city_map.keys()),
                help=(
                    "The training dataset was based on locations in India. "
                    "Select Others when your location is outside the listed "
                    "Indian regions."
                ),
            )

            profession = st.selectbox(
                "Profession",
                options=list(profession_map.keys()),
            )

            degree_field = st.selectbox(
                "Degree Field",
                options=list(degree_field_map.keys()),
            )

        with column_two:

            academic_pressure = st.slider(
                "Academic Pressure",
                min_value=0,
                max_value=5,
                value=3,
                help="0 represents no pressure and 5 represents very high pressure.",
            )

            work_pressure = st.slider(
                "Work Pressure",
                min_value=0,
                max_value=5,
                value=2,
                help="0 represents no pressure and 5 represents very high pressure.",
            )

            study_satisfaction = st.slider(
                "Study Satisfaction",
                min_value=0,
                max_value=5,
                value=3,
                help="0 represents no satisfaction and 5 represents very high satisfaction.",
            )

            job_satisfaction = st.slider(
                "Job Satisfaction",
                min_value=0,
                max_value=5,
                value=3,
                help="0 represents no satisfaction and 5 represents very high satisfaction.",
            )

            financial_stress = st.slider(
                "Financial Stress",
                min_value=0,
                max_value=5,
                value=3,
                help="0 represents no financial stress and 5 represents very high financial stress.",
            )

        with column_three:

            cgpa = st.number_input(
                "CGPA",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.1,
                format="%.2f",
            )

            work_study_hours = st.number_input(
                "Work/Study Hours Per Day",
                min_value=0.0,
                max_value=24.0,
                value=8.0,
                step=0.5,
            )

            sleep_duration = st.selectbox(
                "Sleep Duration",
                options=list(sleep_duration_map.keys()),
            )

            dietary_habits = st.selectbox(
                "Dietary Habits",
                options=list(dietary_habits_map.keys()),
            )

            suicidal_thoughts = st.selectbox(
                "Have You Had Suicidal Thoughts?",
                options=list(yes_no_map.keys()),
            )

            family_mental_illness = st.selectbox(
                "Family History of Mental Illness",
                options=list(yes_no_map.keys()),
            )

        submit_depression = st.form_submit_button(
            "Predict Depression Risk",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # DEPRESSION PREDICTION
    # --------------------------------------------------------

    if submit_depression:

        try:
            input_values = {
                "Gender": gender_map[gender],
                "Age": age,
                "City": city_map[city],
                "Profession": profession_map[profession],
                "Academic Pressure": academic_pressure,
                "Work Pressure": work_pressure,
                "CGPA": cgpa,
                "Study Satisfaction": study_satisfaction,
                "Job Satisfaction": job_satisfaction,
                "Sleep Duration": sleep_duration_map[sleep_duration],
                "Dietary Habits": dietary_habits_map[dietary_habits],
                "Degree_Field": degree_field_map[degree_field],
                "Suicidal Thoughts": yes_no_map[suicidal_thoughts],
                "Work/Study Hours": work_study_hours,
                "Financial Stress": financial_stress,
                "FM Mental Illness": yes_no_map[family_mental_illness],
            }

            # Create a one-row DataFrame.
            depression_input = pd.DataFrame([input_values])

            # Ensure that the columns are arranged in exactly the same
            # order used when the trained model was created.
            depression_input = depression_input[
                list(depression_feature_names)
            ]

            # Apply the saved scaler.
            scaled_depression_input = depression_scaler.transform(
                depression_input
            )

            # Generate prediction.
            depression_prediction = depression_model.predict(
                scaled_depression_input
            )[0]

            # Generate class probabilities.
            depression_probabilities = depression_model.predict_proba(
                scaled_depression_input
            )[0]

            predicted_class_index = list(
                depression_model.classes_
            ).index(depression_prediction)

            prediction_confidence = (
                depression_probabilities[predicted_class_index] * 100
            )

            st.divider()
            st.subheader("Prediction Result")

            if int(depression_prediction) == 1:
                st.error("Higher Depression Risk Detected")

                st.write(
                    """
                    Based on the information entered, the model predicts
                    that the profile belongs to the depressed-risk class.
                    """
                )

            else:
                st.success("Lower Depression Risk Detected")

                st.write(
                    """
                    Based on the information entered, the model predicts
                    that the profile belongs to the non-depressed class.
                    """
                )

            st.metric(
                "Model Confidence",
                f"{prediction_confidence:.2f}%",
            )

            with st.expander("View Model Input Data"):
                display_input = pd.DataFrame(
                    {
                        "Factor": [
                            "Gender",
                            "Age",
                            "Location",
                            "Profession",
                            "Academic Pressure",
                            "Work Pressure",
                            "CGPA",
                            "Study Satisfaction",
                            "Job Satisfaction",
                            "Sleep Duration",
                            "Dietary Habits",
                            "Degree Field",
                            "Suicidal Thoughts",
                            "Work/Study Hours",
                            "Financial Stress",
                            "Family Mental Illness",
                        ],
                        "Value": [
                            gender,
                            age,
                            city,
                            profession,
                            academic_pressure,
                            work_pressure,
                            cgpa,
                            study_satisfaction,
                            job_satisfaction,
                            sleep_duration,
                            dietary_habits,
                            degree_field,
                            suicidal_thoughts,
                            work_study_hours,
                            financial_stress,
                            family_mental_illness,
                        ],
                    }
                )

                st.dataframe(
                    display_input,
                    use_container_width=True,
                    hide_index=True,
                )

            if suicidal_thoughts == "Yes":
                st.warning(
                    """
                    You indicated that you have experienced suicidal thoughts.
                    This machine-learning result is not an emergency or clinical
                    assessment. Please seek immediate support from a qualified
                    professional or local emergency service if you are in danger.
                    """
                )

        except KeyError as error:
            st.error(
                "A required feature is missing or has a different name."
            )
            st.code(str(error))

            st.write("Features expected by the saved model:")

            st.code(
                "\n".join(
                    str(feature)
                    for feature in depression_feature_names
                )
            )

        except ValueError as error:
            st.error(
                "The information could not be processed by the model."
            )
            st.code(str(error))

        except Exception as error:
            st.error(
                "An unexpected error occurred while generating the prediction."
            )
            st.code(str(error))


# ============================================================
# STRESS ASSESSMENT
# ============================================================

elif assessment_type == "Stress Level Assessment":

    st.header("Stress Level Assessment")

    st.caption(
        "Complete all fields and select Predict Stress Level."
    )

    # --------------------------------------------------------
    # CATEGORY ENCODING MAPS
    # --------------------------------------------------------

    mental_health_history_map = {
        "No": 0,
        "Yes": 1,
    }

    blood_pressure_map = {
        "Low": 1,
        "Moderate/Normal": 2,
        "High": 3,
    }

    stress_label_map = {
        0: "Low Stress",
        1: "Medium Stress",
        2: "High Stress",
    }

    # --------------------------------------------------------
    # STRESS INPUT FORM
    # --------------------------------------------------------

    with st.form("stress_assessment_form"):

        column_one, column_two, column_three = st.columns(3)

        with column_one:

            anxiety_level = st.slider(
                "Anxiety Score",
                min_value=0,
                max_value=21,
                value=7,
                help=(
                    "Total score from a standardised anxiety assessment. "
                    "Higher scores indicate more severe anxiety symptoms."
                ),
            )

            self_esteem = st.slider(
                "Self-Esteem Score",
                min_value=0,
                max_value=30,
                value=15,
                help=(
                    "Total score from a standardised self-esteem assessment. "
                    "Higher scores indicate stronger self-esteem."
                ),
            )

            mental_health_history = st.selectbox(
                "Mental Health History",
                options=list(mental_health_history_map.keys()),
                help="Previous mental health condition.",
            )

            depression_score = st.slider(
                "Depression Score",
                min_value=0,
                max_value=27,
                value=9,
                help=(
                    "Total score from the PHQ-9 depression assessment. "
                    "Nine questions are scored from 0 to 3, producing a "
                    "maximum score of 27. Higher scores indicate more severe "
                    "depression symptoms."
                ),
            )

            headache = st.slider(
                "Headache Severity",
                min_value=0,
                max_value=5,
                value=2,
                help="None to severe.",
            )

            blood_pressure = st.selectbox(
                "Blood Pressure",
                options=list(blood_pressure_map.keys()),
                help="Current blood pressure level.",
            )

            sleep_quality = st.slider(
                "Sleep Quality",
                min_value=0,
                max_value=5,
                value=3,
                help="Poor to excellent.",
            )

        with column_two:

            breathing_problem = st.slider(
                "Breathing Problems",
                min_value=0,
                max_value=5,
                value=1,
                help="None to severe.",
            )

            noise_level = st.slider(
                "Noise Level",
                min_value=0,
                max_value=5,
                value=2,
                help="Quiet to noisy.",
            )

            living_conditions = st.slider(
                "Living Conditions",
                min_value=0,
                max_value=5,
                value=3,
                help="Poor to excellent.",
            )

            safety = st.slider(
                "Sense of Safety",
                min_value=0,
                max_value=5,
                value=3,
                help="Unsafe to safe.",
            )

            basic_needs = st.slider(
                "Basic Needs",
                min_value=0,
                max_value=5,
                value=3,
                help="Unmet to fully met.",
            )

            academic_performance = st.slider(
                "Academic Performance",
                min_value=0,
                max_value=5,
                value=3,
                help="Poor to excellent.",
            )

            study_load = st.slider(
                "Study Load",
                min_value=0,
                max_value=5,
                value=3,
                help="Light to heavy.",
            )

        with column_three:

            teacher_student_relationship = st.slider(
                "Teacher–Student Relationship",
                min_value=0,
                max_value=5,
                value=3,
                help="Poor to excellent.",
            )

            future_career_concerns = st.slider(
                "Future Career Concerns",
                min_value=0,
                max_value=5,
                value=3,
                help="Low to high.",
            )

            social_support = st.slider(
                "Social Support",
                min_value=0,
                max_value=3,
                value=2,
                help="None to strong.",
            )

            peer_pressure = st.slider(
                "Peer Pressure",
                min_value=0,
                max_value=5,
                value=2,
                help="Low to high.",
            )

            extracurricular_activities = st.slider(
                "Extracurricular Activity",
                min_value=0,
                max_value=5,
                value=2,
                help="None to very active.",
            )

            bullying = st.slider(
                "Bullying Exposure",
                min_value=0,
                max_value=5,
                value=1,
                help="None to severe.",
            )

        submit_stress = st.form_submit_button(
            "Predict Stress Level",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # STRESS PREDICTION
    # --------------------------------------------------------

    if submit_stress:

        try:
            stress_input_values = {
                "anxiety_level": anxiety_level,
                "self_esteem": self_esteem,
                "mental_health_history": mental_health_history_map[
                    mental_health_history
                ],
                "depression": depression_score,
                "headache": headache,
                "blood_pressure": blood_pressure_map[blood_pressure],
                "sleep_quality": sleep_quality,
                "breathing_problem": breathing_problem,
                "noise_level": noise_level,
                "living_conditions": living_conditions,
                "safety": safety,
                "basic_needs": basic_needs,
                "academic_performance": academic_performance,
                "study_load": study_load,
                "teacher_student_relationship": (
                    teacher_student_relationship
                ),
                "future_career_concerns": future_career_concerns,
                "social_support": social_support,
                "peer_pressure": peer_pressure,
                "extracurricular_activities": (
                    extracurricular_activities
                ),
                "bullying": bullying,
            }

            # Create one-row DataFrame.
            stress_input = pd.DataFrame([stress_input_values])

            # Arrange columns in the exact order used during training.
            stress_input = stress_input[
                list(stress_feature_names)
            ]

            # Apply the saved scaler.
            scaled_stress_input = stress_scaler.transform(
                stress_input
            )

            # Generate prediction.
            stress_prediction = stress_model.predict(
                scaled_stress_input
            )[0]

            # Generate class probabilities.
            stress_probabilities = stress_model.predict_proba(
                scaled_stress_input
            )[0]

            predicted_class_index = list(
                stress_model.classes_
            ).index(stress_prediction)

            stress_confidence = (
                stress_probabilities[predicted_class_index] * 100
            )

            stress_result = stress_label_map[
                int(stress_prediction)
            ]

            st.divider()
            st.subheader("Prediction Result")

            if int(stress_prediction) == 0:
                st.success(stress_result)

            elif int(stress_prediction) == 1:
                st.warning(stress_result)

            else:
                st.error(stress_result)

            st.write(
                f"Based on the information entered, the model predicts "
                f"a **{stress_result.lower()}** classification."
            )

            st.metric(
                "Model Confidence",
                f"{stress_confidence:.2f}%",
            )

            with st.expander("View Model Input Data"):

                stress_display_input = pd.DataFrame(
                    {
                        "Factor": [
                            "Anxiety Score",
                            "Self-Esteem Score",
                            "Mental Health History",
                            "Depression Score",
                            "Headache Severity",
                            "Blood Pressure",
                            "Sleep Quality",
                            "Breathing Problems",
                            "Noise Level",
                            "Living Conditions",
                            "Sense of Safety",
                            "Basic Needs",
                            "Academic Performance",
                            "Study Load",
                            "Teacher–Student Relationship",
                            "Future Career Concerns",
                            "Social Support",
                            "Peer Pressure",
                            "Extracurricular Activity",
                            "Bullying Exposure",
                        ],
                        "Value": [
                            anxiety_level,
                            self_esteem,
                            mental_health_history,
                            depression_score,
                            headache,
                            blood_pressure,
                            sleep_quality,
                            breathing_problem,
                            noise_level,
                            living_conditions,
                            safety,
                            basic_needs,
                            academic_performance,
                            study_load,
                            teacher_student_relationship,
                            future_career_concerns,
                            social_support,
                            peer_pressure,
                            extracurricular_activities,
                            bullying,
                        ],
                    }
                )

                st.dataframe(
                    stress_display_input,
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "This result reflects the machine-learning model prediction "
                "only and should not be interpreted as a clinical diagnosis."
            )

        except KeyError as error:
            st.error(
                "A required stress feature is missing or has a different name."
            )

            st.code(str(error))

            st.write("Features expected by the saved stress model:")

            st.code(
                "\n".join(
                    str(feature)
                    for feature in stress_feature_names
                )
            )

        except ValueError as error:
            st.error(
                "The stress information could not be processed by the model."
            )

            st.code(str(error))

        except Exception as error:
            st.error(
                "An unexpected error occurred while generating the stress "
                "prediction."
            )

            st.code(str(error))