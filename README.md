# Credit Risk Prediction Project

## Problem Statement

This project focuses on utilizing advanced analytics to predict credit risk for commercial banking clients. The aim is to develop a robust machine learning model to assess the creditworthiness of businesses seeking loans or credit facilities. The primary objective is to enable proactive risk management, improve lending decisions, and minimize potential losses by accurately identifying high-risk clients.

## Approach

### Data Collection

The primary challenge was acquiring data as there were limited public datasets available for commercial banking purposes. Eventually, a data source from the U.S. Small Business Administration (SBA) was identified. The SBA provides data publicly, including information on defaulters and non-defaulters through programs like the 7(a) Loan Program.

- [SBA Website](https://www.sba.gov/)
- [7(a) Loan Program](https://www.sba.gov/funding-programs/loans/7a-loans)

The dataset used spans from 2010 to 2019 and contains approximately 5.4 lakh records with 40 features. The original dataset can be downloaded from [here](https://data.sba.gov/dataset/7-a-504-foia), filename: `FOIA - 7(a)(FY2010-FY2019) as of 230930.csv`.

### Data Cleaning

Data cleaning involved removing redundant data, renaming columns, creating relevant columns, and handling null values. Due to limitations in data collection, a significant portion of the data had to be dropped, reducing the dataset to 35k records and 15 features.

### EDA and Feature Engineering

Exploratory Data Analysis (EDA) and feature engineering were performed to understand the data distribution and relationships. Outlier treatment was done using capping methods, and normalization was achieved through scaling techniques.

### Modelling

Given the classification nature of the problem, the following models were chosen:
- Logistic Regression
- Decision Tree Classifier
- Support Vector Classifier
- Random Forest Classifier
- XGBoost Classifier

Evaluation was based on F1 score, considering the balance between precision and recall. The selected model based on F1 score was the XGBoost Classifier.

### Model Refinement

Feature importance analysis revealed that only five features significantly impacted the target variable. The model was rebuilt using only these features:
- RevolverStatus
- GrossApproval
- InitialInterestRate
- SBAGuaranteedApproval
- TermInMonth

### Model Deployment using Streamlit

The final model was deployed using Streamlit. The deployment folder contains the following files:
- `Credit_risk_Prediction_App.py`: Streamlit application file
- `power_transformer.joblib`: Serialized PowerTransformer object for data transformation
- `Xgboost_model.joblib`: Serialized XGBoost model

Required packages to run the application:
- Streamlit
- Joblib
- Pandas
- PowerTransformer

## Requirements

- Python 3.x
- Pip (Python package installer)

Install the required Python packages using the following command:

```bash
pip install streamlit joblib pandas
