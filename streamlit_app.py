# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
# App title
# Set Streamlit page config
st.set_page_config(page_title="Prediction App", layout="wide")

st.title("Program Completion Prediction App")
st.markdown("---")
# Display 3 images in a row
col1, col2, col3 = st.columns(3)
with col1:
    st.image("1.png", use_container_width=True)

with col2:
    st.image("2.png", use_container_width=True)

with col3:
    st.image("3.png", use_container_width=True)

st.markdown("---")
# Load your trained model pipeline
model = joblib.load('final_model_pipeline.pkl')


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
