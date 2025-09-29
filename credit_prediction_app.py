import streamlit as st
import joblib
import pandas as pd

# Load the scaler and model
try:
    scaler = joblib.load('scaler.pkl')
    xgb_model = joblib.load('xgboost_model.pkl')
except FileNotFoundError:
    st.error("Error: scaler.pkl or xgboost_model.pkl not found. Please ensure you have trained and saved the scaler and model.")
    st.stop()

st.title("Credit Risk Prediction App")

st.write("Enter the details below to predict credit risk.")

# Get user input for features
person_age = st.slider("Person Age", min_value=X['person_age'].min(), max_value=X['person_age'].max(), value=X['person_age'].mean())
person_income = st.number_input("Person Income", min_value=X['person_income'].min(), max_value=X['person_income'].max(), value=X['person_income'].mean())
person_emp_length = st.slider("Person Employment Length (Years)", min_value=X['person_emp_length'].min(), max_value=X['person_emp_length'].max(), value=X['person_emp_length'].mean())
loan_amnt = st.number_input("Loan Amount", min_value=X['loan_amnt'].min(), max_value=X['loan_amnt'].max(), value=X['loan_amnt'].mean())
loan_int_rate = st.slider("Loan Interest Rate", min_value=X['loan_int_rate'].min(), max_value=X['loan_int_rate'].max(), value=X['loan_int_rate'].mean())
loan_percent_income = st.slider("Loan Percent of Income", min_value=X['loan_percent_income'].min(), max_value=X['loan_percent_income'].max(), value=X['loan_percent_income'].mean())
cb_person_cred_hist_length = st.slider("Credit History Length (Years)", min_value=X['cb_person_cred_hist_length'].min(), max_value=X['cb_person_cred_hist_length'].max(), value=X['cb_person_cred_hist_length'].mean())

person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
loan_intent = st.selectbox("Loan Intent", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
cb_person_default_on_file = st.selectbox("Default on File", ['Y', 'N'])


# Create a dictionary from user input
user_data = {
    'person_age': person_age,
    'person_income': person_income,
    'person_emp_length': person_emp_length,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_cred_hist_length': cb_person_cred_hist_length,
    'person_home_ownership': person_home_ownership,
    'loan_intent': loan_intent,
    'loan_grade': loan_grade,
    'cb_person_default_on_file': cb_person_default_on_file
}

# Convert user input to a pandas DataFrame
user_df = pd.DataFrame([user_data])

# Apply one-hot encoding to categorical features - ensure columns match training data
# Need to handle potential missing columns in user_df after one-hot encoding
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
user_df_encoded = pd.get_dummies(user_df, columns=categorical_cols, drop_first=True)

# Add dummy columns that might be missing in the user input but were present in the training data
# and ensure the order of columns matches the training data (X)
for col in X.columns:
    if col not in user_df_encoded.columns:
        user_df_encoded[col] = False # or 0 depending on how dummies were created

# Ensure the order of columns is the same as in the training data
user_df_encoded = user_df_encoded[X.columns]

# Scale the user input
user_input_scaled = scaler.transform(user_df_encoded)

# Make prediction
if st.button("Predict Credit Risk"):
    prediction = xgb_model.predict(user_input_scaled)
    prediction_proba = xgb_model.predict_proba(user_input_scaled)[:, 1] # Probability of default

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"High Credit Risk (Probability of Default: {prediction_proba[0]:.2f})")
    else:
        st.success(f"Low Credit Risk (Probability of Default: {prediction_proba[0]:.2f})")

st.write("""
**Instructions to run the app:**
1. Save the code above as a Python file (e.g., `credit_prediction_app.py`).
2. Open your terminal or command prompt.
3. Navigate to the directory where you saved the file.
4. Run the command: `streamlit run credit_prediction_app.py`
""")
