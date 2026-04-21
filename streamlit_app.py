
import streamlit as st
import pandas as pd
import pickle
import gzip
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Salary Prediction App", layout="wide")

st.title("Data Science Salary Predictor")
st.write("Enter the details below to predict the salary for a Data Science job.")

# --- Load the Model ---
# Ensure 'best_model.pkl.gz' is in the same directory as your app on Streamlit Cloud
try:
    with gzip.open('best_model.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    st.success("Regression model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Load Original Data for Encoders ---
# Ensure 'Salary_Dataset_DataScienceLovers.csv' is in the same directory
try:
    original_df = pd.read_csv('Salary_Dataset_DataScienceLovers.csv')
    # Apply the same preprocessing steps as training for consistency
    original_df.dropna(inplace=True)

    le_mappers = {}
    categorical_cols = original_df.select_dtypes(include='object').columns
    for col in categorical_cols:
        le = LabelEncoder()
        # Convert column to string type before fitting to handle mixed types gracefully
        le.fit(original_df[col].astype(str)) 
        le_mappers[col] = le

    st.success("Preprocessing data loaded successfully!")
except Exception as e:
    st.error(f"Error loading original data for encoders: {e}")
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
    # Check if 'Employment Status' and 'Job Roles' are present in the loaded encoders
    # This assumes these columns exist in your original dataset and were encoded.
    if 'Employment Status' in le_mappers:
        employment_status_options = le_mappers['Employment Status'].classes_.tolist()
        employment_status = st.selectbox("Employment Status", options=employment_status_options)
    else:
        employment_status = 'Full-time' # Default or handle as appropriate
        st.warning("Employment Status column not found for encoding. Using default.")

    if 'Job Roles' in le_mappers:
        job_roles_options = le_mappers['Job Roles'].classes_.tolist()
        job_roles = st.selectbox("Job Roles", options=job_roles_options)
    else:
        job_roles = 'Data Scientist' # Default or handle as appropriate
        st.warning("Job Roles column not found for encoding. Using default.")

# --- Make Prediction ---
if st.button("Predict Salary"): # Changed button text
    try:
        # Preprocess input
        company_name_encoded = le_mappers['Company Name'].transform([company_name])[0]
        job_title_encoded = le_mappers['Job Title'].transform([job_title])[0]
        location_encoded = le_mappers['Location'].transform([location])[0]

        # Handle optional columns based on their presence in le_mappers
        employment_status_encoded = le_mappers['Employment Status'].transform([employment_status])[0] if 'Employment Status' in le_mappers else 0 # Default to 0 if not found
        job_roles_encoded = le_mappers['Job Roles'].transform([job_roles])[0] if 'Job Roles' in le_mappers else 0 # Default to 0 if not found

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

        # Ensure columns are in the same order as X_train during training
        # Dynamically determine expected columns from the model's training features (X.columns in the notebook)
        # For this example, assuming the features are in a fixed order as observed in the notebook state:
        expected_columns = ['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles']
        input_data = input_data[expected_columns]

        # Make prediction
        prediction = model.predict(input_data)

        # Display result as a continuous salary value
        st.subheader("Predicted Salary:")
        st.success(f"The predicted salary is: **${prediction[0]:,.2f}**") # Formatting as currency

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

