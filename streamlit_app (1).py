
import streamlit as st
import pandas as pd
import pickle
import gzip
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Salary Prediction App", layout="wide")

st.title("Data Science Salary Category Predictor")
st.write("Enter the details below to predict the salary category (Low, Medium, High) for a Data Science job.")

# --- Load the Model ---
# Ensure 'best_model.pkl.gz' is in the same directory as your app on Streamlit Cloud
try:
    with gzip.open('best_model.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Load Original Data for Encoders and Qcut Bins ---
# Ensure 'Salary_Dataset_DataScienceLovers.csv' is in the same directory
try:
    original_df = pd.read_csv('Salary_Dataset_DataScienceLovers.csv')
    # Apply the same preprocessing steps as training for consistency
    original_df.dropna(inplace=True)

    le_mappers = {}
    categorical_cols = original_df.select_dtypes(include='object').columns
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(original_df[col]) # Fit on the original column to get the mapping
        le_mappers[col] = le

    # Get qcut bins for salary categorization (though RandomForestClassifier predicts category directly)
    # This part is primarily for consistency if other models were used or if direct salary prediction was needed
    _, salary_bins = pd.qcut(original_df['Salary'], q=3, labels=['Low', 'Medium', 'High'], retbins=True)
    salary_labels = ['Low', 'Medium', 'High']

    st.success("Preprocessing data loaded successfully!")
except Exception as e:
    st.error(f"Error loading original data for encoders/bins: {e}")
    st.stop()

# --- Input Features ---
col1, col2, col3 = st.columns(3)

with col1:
    rating = st.number_input("Rating (e.g., 3.8)", min_value=1.0, max_value=5.0, value=3.8, step=0.1)
    salaries_reported = st.number_input("Salaries Reported", min_value=1, value=3, step=1)

with col2:
    company_name_options = le_mappers['Company Name'].classes_.tolist()
    company_name = st.selectbox("Company Name", options=company_name_options)
    job_title_options = le_mappers['Job Title'].classes_.tolist()
    job_title = st.selectbox("Job Title", options=job_title_options)

with col3:
    location_options = le_mappers['Location'].classes_.tolist()
    location = st.selectbox("Location", options=location_options)
    employment_status_options = le_mappers['Employment Status'].classes_.tolist()
    employment_status = st.selectbox("Employment Status", options=employment_status_options)
    job_roles_options = le_mappers['Job Roles'].classes_.tolist()
    job_roles = st.selectbox("Job Roles", options=job_roles_options)

# --- Make Prediction ---
if st.button("Predict Salary Category"):
    try:
        # Preprocess input
        company_name_encoded = le_mappers['Company Name'].transform([company_name])[0]
        job_title_encoded = le_mappers['Job Title'].transform([job_title])[0]
        location_encoded = le_mappers['Location'].transform([location])[0]
        employment_status_encoded = le_mappers['Employment Status'].transform([employment_status])[0]
        job_roles_encoded = le_mappers['Job Roles'].transform([job_roles])[0]

        # Create DataFrame for prediction
        input_data = pd.DataFrame([{
            'Rating': rating,
            'Company Name': company_name_encoded,
            'Job Title': job_title_encoded,
            'Salaries Reported': salaries_reported,
            'Location': location_encoded,
            'Employment Status': employment_status_encoded,
            'Job Roles': job_roles_encoded
        }])

        # Ensure columns are in the same order as X_train_clf during training
        # This order was: ['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles']
        expected_columns = ['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles']
        input_data = input_data[expected_columns]

        # Make prediction
        prediction = model.predict(input_data)

        # Display result
        st.subheader("Predicted Salary Category:")
        st.success(f"The predicted salary category is: **{prediction[0]}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

