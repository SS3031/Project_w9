import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder



#load csv
df_reordered = pd.read_csv('C:/Users/sophi/Ironhack/Ironhack/Ironhack_prework_Jupyter/Week9/csv/df_reordered.csv')

# Load the scaler
filename = "C:/Users/sophi/Ironhack/Ironhack/Ironhack_prework_Jupyter/Week9/scalers/standard_scaler.pkl"
with open(filename, "rb") as file:
    scaler = pickle.load(file)

# Load the encoder
filename = "C:/Users/sophi/Ironhack/Ironhack/Ironhack_prework_Jupyter/Week9/encoders/one_hot_encoder.pkl"
with open(filename, "rb") as file:
    encoder = pickle.load(file)


# Load the trained RandomForestClassifier model
filename = "C:/Users/sophi/Ironhack/Ironhack/Ironhack_prework_Jupyter/Week9/models/random_forest.pkl"
with open(filename, "rb") as file:
   model = pickle.load(file)





# Streamlit app UI
st.write("""
    # Hello, Welcome to the credit risk prediction app
    This app predicts the risk involved in granting credit to clients
    """)

st.sidebar.header('User Input Parameters')

# Define the options for credit history and purpose
def user_input_features():
    credit_history_options = ['critical account/other credits existing',
                         'existing credits paid back duly till now',
                         'delay in paying off in the past',
                         'no credits taken/all credits paid back duly',
                         'all credits at this bank paid back duly']
    purpose_options = ['domestic appliances', 'retraining', 'radio/television', 'car (new)',
                   'car (used)', 'others', 'repairs', 'education', 'furniture/equipment', 'business']               
    savings_options = ['unknown/no savings account', '... < 100 DM',
       '500 <= ... < 1000 DM', '... >= 1000 DM', '100 <= ... < 500 DM']
    housing_options = ['own', 'for free', 'rent']
    other_debtors_options = ['none', 'guarantor', 'co-applicant']
    property_options = ['real estate', 'building society savings agreement/life insurance',
       'unknown/no property', 'car or other']
    other_installment_plan_options = ['none', 'bank', 'stores']
    job_options = ['skilled employee/official', 'unskilled - resident',
       'management/self-employed/highly qualified employee/officer',
       'unemployed/unskilled - non-resident']
    telephone_options = ['yes', 'no']
    foreign_worker_options = ['yes', 'no']
    age_group_options = ['Retired', 'Young_professionals', 'Older_professionals', 'Seniors', 'Middle-aged_professionals']
    sex_options = ['female', 'male']
    marital_status_options = ['single', 'divorced/separated/married', 'divorced/separated', 'married/widowed']

def user_input_features():
    duration = st.number_input('Duration')
    st.write('The current number is ', duration)
    credit_history = st.sidebar.selectbox('Credit History', credit_history_options)
    purpose = st.sidebar.selectbox('Purpose', purpose_options)
    amount= ['Amount', amount_options]
    savings = st.sidebar.selectbox('Savings', savings_options)
    installment_rate = ['installment_rate', installment_rate_options]
    housing = st.sidebar.selectbox('housing', housing_options)
    other_debtors = st.sidebar.selectbox('other_debtors', other_debtors_options)
    property = st.sidebar.selectbox('property', property_options)
    other_installment_plans = st.sidebar.selectbox('other_installment_plans', other_installment_plan_options)
    job = st.sidebar.selectbox('job', job_options)
    people_liable=['People_liable', people_liable_options]
    telephone = st.sidebar.selectbox('telephone', telephone_options)
    foreign_worker = st.sidebar.selectbox('foreign_worker', foreign_worker_options)
    age_group = st.sidebar.selectbox('age_group', age_group_options)
    sex = st.sidebar.selectbox('Sex', sex_options )
    marital_status = st.sidebar.selectbox('marital_status', marital_status_options)
    data = {'duration' : [duration],
    'credit_history':[credit_history],
    'purpose': [purpose],
    'amount' :[amount],
    'savings' :[savings],
    'installment_rate': [installment_rate],
    'housing': [housing],
    'other_debtors': [other_debtors],
    'property':[property],
    'other_installment_plans':[other_installment_plans],
    'job': [job],
    'people_liable': [people_liable],
    'telephone' : [telephone],
    'foreign_worker': [foreign_worker],
    'age_group':[age_group],
    'sex': [sex],
    'marital_status': [marital_status]}
    features = pd.DataFrame(data)
    return features

df = user_input_features()


# Apply the encoder, scaler, 






st.subheader('Final user input choice')
st.write(df)

df_input = df.copy()
# Prepare your data
selected_categorical_columns = ['credit_history',  'purpose', 'savings', 'other_debtors',  'property',
       'other_installment_plans', 'housing', 'job', 'telephone', 'foreign_worker', 'sex',
        'age_group', 'marital_status']

selected_numerical_columns = ['installment_rate', 'duration', 'amount', 'number_credits', 'people_liable' ]




df_input_cat = df_input[selected_categorical_columns]

df_input_num = df_input[selected_numerical_columns]

df_input_cat_encoded_np = encoder.transform(df_input_cat)


#X_train_cat_encoded_np # X_train_cat.shape
df_input_cat = pd.DataFrame(df_input_cat_encoded_np, columns=encoder.get_feature_names_out(), index=df_input_cat.index) # [0,1]


df_input = pd.concat([df_input_num, df_input_cat], axis=1)



df_input_trans_np = scaler.transform(df_input)


df_input_trans = pd.DataFrame(df_input_trans_np, columns=df_input.columns, index=df_input.index) 




# Predict on the testing set
input_pred_rf = model.predict(df_input_trans)

# Evaluate Random Forest model
accuracy_rf = accuracy_score(df_input_target, input_pred_rf)

classification_rep_rf = classification_report(df_input_target, input_pred_rf)

st.write("Random Forest Classifier:")
st.write(f"Accuracy: {accuracy_rf}")

st.write("Classification Report:")
st.write(classification_rep_rf)
st.write("\n")

if input_pred_rf == 1:
    st.write("Good candidate")
else:
    st.write("Not a good candidate")


