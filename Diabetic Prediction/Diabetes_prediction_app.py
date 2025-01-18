import streamlit as st
import pandas as pd
from joblib import load

# Load the model and the scaler
gb_model = load('gb_model.pkl')
scaler = load('scaler.pkl')
isolation_forest_model = load('isolation_forest_model.pkl')

def remove_outliers(input_data):
    outliers = isolation_forest_model.predict(input_data)
    input_data_filtered = input_data[outliers == 1]
    return input_data_filtered

def make_prediction(genhlth, highbp, diffwalk, bmi, highchol, age, heartdisease, physhlth, phyactivity, education,
                    income):
    input_data_filtered = pd.DataFrame({
        'GenHlth': [genhlth],
        'HighBP': [highbp],
        'DiffWalk': [diffwalk],
        'BMI': [bmi],
        'HighChol': [highchol],
        'Age': [age],
        'HeartDiseaseorAttack': [heartdisease],
        'PhysHlth': [physhlth],
        'PhysActivity': [phyactivity],
        'Education': [education],
        'Income': [income]
    })
    # Scaling the filtered input data
    input_scaled = scaler.transform(input_data_filtered)
    # Making the prediction using the gradient boosting model
    prediction = gb_model.predict(input_scaled)
    return prediction
# Streamlit app
def main():
    st.title("Diabetes Prediction App")
    # Sidebar options
    st.sidebar.header("Navigation")
    menu = ["Home", "Diabetes Prediction"]
    choice = st.sidebar.selectbox("Select Option", menu)

    if choice == "Home":
        render_home()
    elif choice == "Diabetes Prediction":
        render_diabetes_prediction()

def render_home():
    st.subheader("Home")
    st.write("Welcome to the Diabetes Prediction App!")
    st.write("This app is designed to help you predict whether you are likely to have diabetes or prediabetes based on your health-related information. By entering specific parameters related to your general health, blood pressure, physical activity, education level, and more, the app utilizes a machine learning model to provide you with an estimated prediction.")

    st.write("Here's how to use the app:")
    st.write("1. Go to the 'Diabetes Prediction' section from the sidebar.")
    st.write("2. Enter your health-related information in the input fields.")
    st.write("3. Click the 'Predict' button to obtain the prediction result.")

    st.write( "Please note that the predictions made by this app are estimates based on the provided information and the machine learning model. It is always recommended to consult with a healthcare professional for a comprehensive evaluation and diagnosis.")

    st.write("Feel free to navigate through the app using the sidebar on the left to explore more about diabetes prediction or make predictions using your health data.")

def render_diabetes_prediction():
        st.subheader("Diabetes Prediction")

        # Create input fields for user input
        genhlth_mapping = {"1: Excellent": 1, "2: Very Good": 2, "3: Good": 3, "4: Fair": 4, "5: Poor": 5}
        genhlth = st.selectbox("General Health", list(genhlth_mapping.keys()))

        highbp = st.selectbox("High Blood Pressure", [0, 1])

        diffwalk = st.selectbox("Difficulty Walking", [0, 1])

        bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0)

        highchol = st.selectbox("High Cholesterol", [0, 1])

        age_mapping = {"18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
                       "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
                       "70-74": 11, "75-79": 12, "80 or older": 13}
        age = st.selectbox("Age", list(age_mapping.keys()))

        heartdisease = st.selectbox("Heart Disease or Heart Attack", [0, 1])

        physhlth = st.number_input("Physical Health", min_value=0, max_value=30, value=15)

        phyactivity = st.selectbox("Physical Activity", [0, 1])

        education_mapping = {
            "1: Never attended school or only kindergarten": 1,
            "2: Grades 1 through 8 (Elementary)": 2,
            "3: Grades 9 through 11 (Some high school)": 3,
            "4: Grade 12 or GED (High school graduate)": 4,
            "5: College 1 year to 3 years (Some college or technical school)": 5,
            "6: College 4 years or more (College graduate)": 6
        }
        education = st.selectbox("Education Level", list(education_mapping.keys()))

        income_mapping = {
            "Less than $10,000": 1,
            "Less than $15,000": 2,
            "Less than $20,000": 3,
            "Less than $25,000": 4,
            "Less than $35,000": 5,
            "Less than $50,000": 6,
            "Less than $75,000": 7,
            "$75,000 or more": 8
        }
        income = st.selectbox("Income", list(income_mapping.keys()))

        if st.button("Predict"):
            # Map categorical options to numerical values for prediction
            genhlth = genhlth_mapping[genhlth]
            age = age_mapping[age]
            education = education_mapping[education]
            income = income_mapping[income]



        # Make prediction
            prediction = make_prediction(genhlth, highbp, diffwalk, bmi, highchol, age, heartdisease,
                                                           physhlth, phyactivity, education, income)

    # Display the prediction result
            st.subheader("Prediction Result")
            if prediction == 0:
              st.write("The individual is predicted to have no diabetes.")
            else:
              st.write("The individual is predicted to have diabetes.")


# Run the app
if __name__ == "__main__":
    main()