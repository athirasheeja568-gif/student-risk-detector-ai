import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student At-Risk Detector", layout="centered")

st.title("Student At-Risk Detector")
st.success("AI-powered early warning system for student performance")
st.write("Upload a dataset and predict whether a student is at academic risk.")

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    required_columns = ["Hours", "Attendance", "Assignments", "RiskLevel"]

    # Check required columns
    if all(col in df.columns for col in required_columns):
        # Prepare data
        X = df[["Hours", "Attendance", "Assignments"]]
        y = df["RiskLevel"]

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y_encoded)

        st.subheader("Enter Student Details")

        hours = st.slider("Study Hours", 1, 10, 4)
        attendance = st.slider("Attendance (%)", 40, 100, 75)
        assignments = st.slider("Assignment Score", 30, 100, 65)

        if st.button("Check Risk Level"):
            input_data = pd.DataFrame({
                "Hours": [hours],
                "Attendance": [attendance],
                "Assignments": [assignments]
            })

            prediction_encoded = model.predict(input_data)[0]
            prediction = label_encoder.inverse_transform([prediction_encoded])[0]

            st.subheader("Prediction Result")
            st.success(f"Risk Level: {prediction}")

            st.subheader("Recommendations")

            if prediction == "High":
                st.error("This student is at high academic risk.")
                st.write("- Increase daily study hours")
                st.write("- Improve attendance immediately")
                st.write("- Submit assignments on time")

            elif prediction == "Medium":
                st.warning("This student is at medium academic risk.")
                st.write("- Be more consistent with studying")
                st.write("- Improve assignment performance")
                st.write("- Maintain better class attendance")

            else:
                st.info("This student is at low academic risk.")
                st.write("- Keep up the good work")
                st.write("- Stay consistent")
                st.write("- Continue regular study habits")
    else:
        st.error("Dataset must contain these columns: Hours, Attendance, Assignments, RiskLevel")

else:
    st.info("Please upload a CSV or Excel file to continue.")