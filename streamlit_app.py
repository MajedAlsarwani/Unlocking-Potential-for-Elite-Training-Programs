# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
# App title
# Set Streamlit page config
st.set_page_config(page_title="Prediction App", layout="wide")

st.title("Program Completion Prediction for Tuwaiq Academy")
st.markdown("---")



st.markdown(" Note: test data should be in the same format as the training data.")
# Load your trained model pipeline
model = joblib.load('final_model_pipeline.pkl')
encoders = joblib.load('label_encoders.pkl')

st.sidebar.header("Upload Your Test File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Make predictions
    if st.button("Predict"):
        try:
            probs = model.predict_proba(df)[:, 1]  # Probability of class 1 (not completed)
            predictions = (probs > 0.38).astype(int)  # Use your tuned threshold

            df_result = df.copy()
            df_result['Prediction'] = predictions
            df_result['Probability_Not_Completed'] = probs

            st.subheader("Prediction Results")
            st.dataframe(df_result)

            # Download button
            csv = df_result.to_csv(index=False)
            st.download_button("Download Results", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {e}")

# === Manual Input Prediction Section ===
st.markdown("## Or Manually Enter Values for Prediction")

with st.form("manual_input_form"):
    st.markdown("### Enter Input Values")

    age = st.number_input("Age", min_value=10, max_value=70, value=25)
    gender = st.selectbox("Gender", options=["ذكر", "أنثى"])
    home_region = st.selectbox(
    "Home Region", 
    options=[
        "منطقة الرياض",
        "منطقة مكة المكرمة",
        "المنطقة الشرقية",
        "منطقة المدينة المنورة",
        "منطقة عسير",
        "منطقة القصيم",
        "منطقة جازان",
        "منطقة تبوك",
        "منطقة الباحة",
        "منطقة حائل",
        "منطقة نجران",
        "منطقة الحدود الشمالية",
        "منطقة الجوف"
        ]
    ) # Update list as needed
    home_city = st.text_input("Home City", value="الرياض")
    main_category = st.selectbox(
        "Program Main Category Code", 
        options=[
            "CAUF",
            "PCRF",
            "APMR",
            "TOSL",
            "GRST",
            "ABIR",
            "INFA",
            "SERU",
            "DTFH",
            "QWLM"
        ]
    ) # Extend list
    sub_category = st.selectbox(
    "Program Sub Category Code",
    options=[
        "SWPS",
        "PCRF",
        "SRTA",
        "INFA",
        "TOSL",
        "APMR",
        "CAUF",
        "CRDP",
        "ERST",
        "KLTM",
        "ABIR",
        "QTDY",
        "ASCW",
        "DTFH",
        "QWLM"
        ]
    )

    skill_level = st.selectbox("Program Skill Level", options=["غير معروف", "متوسط", "مبتدئ", "متقدم"])
    presentation_method = st.selectbox("Program Presentation Method", options=["حضوري", "عن بعد"])

    program_days = st.number_input("Program Days", min_value=1, max_value=300, value=10)
    completed_degree = st.selectbox("Completed Degree", options=["نعم", "لا"])
    education_level = st.selectbox("Level of Education", options=["ثانوي", "الدبلوم", "البكالوريوس", "الماجستير", "الدكتوراه"])
    employment_status = st.selectbox("Employment Status", options=["طالب", "خريج", "موظف", "غير موظف", "غير معروف"])

    unified_score = st.number_input("Unified Score Percentage", min_value=0.0, max_value=100.0, value=75.0)
    college_filled = st.selectbox("College (Filled)", options=[
    "تكنولوجيا الاتصالات والمعلومات",
    "كلية أخرى",
    "كلية العلوم",
    "الأعمال والإدارة والقانون",
    "العلوم الطبيعية والرياضيات والإحصاء",
    "إدارة الأعمال",
    "الهندسة والتصنيع والبناء",
    "الفنون والعلوم الإنسانية",
    "العلوم الاجتماعية والصحافة والإعلام",
    "الطب",
    "التعليم",
    "التربية",
    "الآداب والترجمة",
    "الصحة والرفاة",
    "الهندسة",
    "القانون أو الشريعة",
    "البرامج والمؤهلات العامة"
])
    total_registration = st.number_input("Total Registration", min_value=1, max_value=110, value=5)

    submit_button = st.form_submit_button("Predict Manually")

    if submit_button:
        try:
            input_dict = {
                "Age": [age],
                "Gender": [gender],
                "Home Region": [home_region],
                "Home City": [home_city],
                "Program Main Category Code": [main_category],
                "Program Sub Category Code": [sub_category],
                "Program Skill Level": [skill_level],
                "Program Presentation Method": [presentation_method],
                "Program Days": [program_days],
                "Completed Degree": [completed_degree],
                "Level of Education": [education_level],
                "Employment Status": [employment_status],
                "Unified_Score_Percentage": [unified_score],
                "College_Filled": [college_filled],
                "Total Regestration": [total_registration]
            }

            input_df = pd.DataFrame(input_dict)
            for col in input_df.select_dtypes(include='object').columns:
                if col in encoders:
                    input_df[col] = encoders[col].transform(input_df[col])
            prob = model.predict_proba(input_df)[0][1]
            prediction = int(prob > 0.38)

            st.success(f"**Prediction:** {'he/she will Not Complete the program' if prediction == 1 else 'he/she will Complete the program'}")
            st.info(f"Probability of Not Completing: {prob:.2f}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
