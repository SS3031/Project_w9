import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from pathlib import Path

# set parent path of the script
script_dir = Path(__file__).parents[0]

#load csv
#df_reordered = pd.read_csv('C:/Users/sophi/Ironhack/Ironhack/Ironhack_prework_Jupyter/Week9/csv/df_reordered.csv')

# Load the scaler
filename = str(script_dir)+"/scalers/standard_scaler.pkl"
with open(filename, "rb") as file:
    scaler = pickle.load(file)

# Load the encoder
filename = str(script_dir)+"/encoders/one_hot_encoder.pkl"
with open(filename, "rb") as file:
    encoder = pickle.load(file)


# Load the trained RandomForestClassifier model
filename = str(script_dir)+"/models/random_forest.pkl"
with open(filename, "rb") as file:
   model = pickle.load(file)





# Streamlit app UI
st.write("""
    # Hello, Welcome to the credit risk prediction app
    This app predicts the risk involved in granting credit to clients
    """)

st.sidebar.header('User Input Parameters')

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
    sex_options = ['male','female']
    marital_status_options = ['single', 'divorced/separated/married', 'divorced/separated', 'married/widowed']
    
    duration = st.sidebar.number_input('Duration')#################
    #st.write('The current number is ', duration)
    credit_history = st.sidebar.selectbox('Credit History', credit_history_options)
    purpose = st.sidebar.selectbox('Purpose', purpose_options)
    amount= st.sidebar.number_input('amount')############
    savings = st.sidebar.selectbox('Savings', savings_options)
    installment_rate = st.sidebar.number_input('installment_rate')#################
    housing = st.sidebar.selectbox('housing', housing_options)
    other_debtors = st.sidebar.selectbox('other_debtors', other_debtors_options)
    property = st.sidebar.selectbox('property', property_options)
    other_installment_plans = st.sidebar.selectbox('other_installment_plans', other_installment_plan_options)
    job = st.sidebar.selectbox('job', job_options)
    people_liable=st.sidebar.number_input('people_liable')
    telephone = st.sidebar.selectbox('telephone', telephone_options)
    foreign_worker = st.sidebar.selectbox('foreign_worker', foreign_worker_options)
    age_group = st.sidebar.selectbox('age_group', age_group_options)
    sex = st.sidebar.selectbox('Sex', sex_options )
    marital_status = st.sidebar.selectbox('marital_status', marital_status_options)
    number_credits = st.sidebar.number_input('number_credits')#################

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
    'marital_status': [marital_status],
    'number_credits':[number_credits]}
    features = pd.DataFrame(data)
    return features

df = user_input_features()
#st.dataframe(df)


# Apply the encoder, scaler, 




st.subheader('Final user input choice')
st.dataframe(df)

df_input = df.copy()
# Prepare your data
selected_categorical_columns = ['credit_history',  'purpose', 'savings', 'other_debtors',  'property',
       'other_installment_plans', 'housing', 'job', 'telephone', 'foreign_worker', 'sex',
        'age_group', 'marital_status']

selected_numerical_columns = ['installment_rate', 'duration', 'amount', 'number_credits', 'people_liable' ]




df_input_cat = df_input[selected_categorical_columns]

df_input_num = df_input[selected_numerical_columns]

#st.write("df_input_cat",df_input_cat)

df_input_cat_encoded_np = encoder.transform(df_input_cat)


#X_train_cat_encoded_np # X_train_cat.shape
df_input_cat = pd.DataFrame(df_input_cat_encoded_np, columns=encoder.get_feature_names_out(), index=df_input_cat.index) # [0,1]


df_input = pd.concat([df_input_num, df_input_cat], axis=1)



df_input_trans_np = scaler.transform(df_input)


df_input_trans = pd.DataFrame(df_input_trans_np, columns=df_input.columns, index=df_input.index) 




# Predict on the testing set
input_pred_rf = model.predict(df_input_trans)



if input_pred_rf == 1:
    st.write("Good candidate")
else:
    st.write("Not a good candidate")

