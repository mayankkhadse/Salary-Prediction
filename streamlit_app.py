import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load('linear_regression_model.pkl')

# Load dataset
df = pd.read_csv('Salary_Dataset_DataScienceLovers.csv')
df.columns = df.columns.str.strip()

# Encoders (NO PKL NEEDED)
le_company = LabelEncoder().fit(df['Company Name'])
le_job = LabelEncoder().fit(df['Job Title'])
le_loc = LabelEncoder().fit(df['Location'])
le_emp = LabelEncoder().fit(df['Employment Status'])
le_role = LabelEncoder().fit(df['Job Roles'])

st.title("Salary Prediction App")

# Dropdowns
company = st.selectbox("Company", le_company.classes_)
job = st.selectbox("Job Title", le_job.classes_)
location = st.selectbox("Location", le_loc.classes_)
employment = st.selectbox("Employment Status", le_emp.classes_)
role = st.selectbox("Job Role", le_role.classes_)

rating = st.slider("Rating", 0.0, 5.0, 3.5)
salary_rep = st.number_input("Salaries Reported", 1, 100, 1)

if st.button("Predict"):

    input_data = pd.DataFrame([{
        'Rating': rating,
        'Company Name': le_company.transform([company])[0],
        'Job Title': le_job.transform([job])[0],
        'Salaries Reported': salary_rep,
        'Location': le_loc.transform([location])[0],
        'Employment Status': le_emp.transform([employment])[0],
        'Job Roles': le_role.transform([role])[0]
    }])

    pred = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: {pred:,.2f} INR")
