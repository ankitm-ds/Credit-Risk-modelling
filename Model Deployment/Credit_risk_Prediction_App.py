import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import PowerTransformer
# Load the power transformer
preprocessor = joblib.load('power_transformer.joblib')
# Load the XGBoost model
model = joblib.load('xgboost_model.joblib')
# Function to make predictions
def predict(input_data):
    # Apply the same power transformation to the input data
    input_df = pd.DataFrame([input_data])
    input_transformed = preprocessor.transform(input_df)
    # Make predictions using the XGBoost model
    prediction = model.predict(input_transformed)
    return prediction[0]
# Helper function to convert values to million-dollar units
def convert_to_million(value):
    return value / 1_000_000.0
# Streamlit app
def main():
    st.title('XGBoost Classifier Deployment')
    # Add input fields
    term_in_months = st.slider('Term in Months', min_value=0, max_value=324, value=119)
    sbaguaranteed_approval = st.slider('SBA Guaranteed Approval (Million $)', min_value=convert_to_million(1000.0), max_value=convert_to_million(4500000.0), value=convert_to_million(75000.0))
    initial_interest_rate = st.slider('Interest Rate', min_value=1.0, max_value=13.5, value=7.25)
    gross_approval = st.slider('Gross Approval (Million $)', min_value=convert_to_million(2000.0), max_value=convert_to_million(5000000.0), value=convert_to_million(115000.0))
    revolver_status = st.checkbox('Revolver Status ', value=False)
    # Button to make predictions
    if st.button('Predict'):
        # Check if SBAGuaranteedApproval is less than GrossApproval
        if sbaguaranteed_approval * 1_000_000.0 < gross_approval * 1_000_000.0:
            input_data = {
                'TermInMonths': term_in_months,
                'SBAGuaranteedApproval': sbaguaranteed_approval * 1_000_000.0,  # Convert back to original unit
                'InitialInterestRate': initial_interest_rate,
                'GrossApproval': gross_approval * 1_000_000.0,  # Convert back to original unit
                'RevolverStatus': 1 if revolver_status else 0  # Convert checkbox value to 1 or 0
            }
            prediction = predict(input_data)
            # st.success(f'The prediction is: {prediction}')
            # Display the prediction result
            if prediction == 1:
                st.success('Application Approved!')
            else:
                st.error('Application Rejected.')
        else:
                st.warning('SBAGuaranteedApproval should be less than GrossApproval. Please adjust the values.')
if __name__ == '__main__':
    main()