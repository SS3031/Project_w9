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
#df_reordered = pd.read_csv('C:/Users/sophi/Ironhack/Ironhack/Ironhack_prework_Jupyter/Week9/csv/df_reordered1.csv')

# Load the scaler
filename = str(script_dir)+"/scalers/standard_scaler1.pkl"
with open(filename, "rb") as file:
    scaler = pickle.load(file)

# Load the encoder
filename = str(script_dir)+"/encoders/one_hot_encoder1.pkl"
with open(filename, "rb") as file:
    encoder = pickle.load(file)


# Load the trained RandomForestClassifier model
filename = str(script_dir)+"/models/random_forest1.pkl"
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
                         'no existing credit history',
                         'all credits at this bank paid back duly']
    purpose_options = ['domestic appliances', 'retraining', 'radio/television', 'car (new)',
                   'car (used)', 'others', 'repairs', 'education', 'furniture/equipment', 'business']               
    housing_options = ['own', 'for free', 'rent']
    property_options = ['real estate', 'building society savings agreement/life insurance',
       'unknown/no property', 'car or other']
    job_options = ['skilled employee/official', 'unskilled - resident',
       'management/self-employed/highly qualified employee/officer',
       'unemployed/unskilled - non-resident']
    age_group_options = ['Retired', 'Young_professionals', 'Older_professionals', 'Seniors', 'Middle-aged_professionals']
    sex_options = ['male','female']
    marital_status_options = ['single', 'married', 'divorced/separated', 'widowed']
    
    duration = st.sidebar.number_input('Duration')#################
    #st.write('The current number is ', duration)
    credit_history = st.sidebar.selectbox('Credit History', credit_history_options)
    purpose = st.sidebar.selectbox('Purpose', purpose_options)
    amount= st.sidebar.number_input('amount')############
    housing = st.sidebar.selectbox('housing', housing_options)
    property = st.sidebar.selectbox('property', property_options)
    job = st.sidebar.selectbox('job', job_options)
    age_group = st.sidebar.selectbox('age_group', age_group_options)
    sex = st.sidebar.selectbox('Sex', sex_options )
    marital_status = st.sidebar.selectbox('marital_status', marital_status_options)
    number_credits = st.sidebar.number_input('number_credits')#################

    data = {'duration' : [duration],
    'credit_history':[credit_history],
    'purpose': [purpose],
    'amount' :[amount],
    'housing': [housing],
    'property':[property],
    'job': [job],
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
selected_categorical_columns = ['credit_history',  'purpose',  'property',
       'housing', 'job', 'sex',
        'age_group', 'marital_status']

selected_numerical_columns = ['duration', 'amount', 'number_credits' ]




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




def predict(input_pred_rf):
    if input_pred_rf == 1:
        return "Good candidate"
    else:
        return "Not a good candidate"

# Streamlit UI
st.title("Candidate Prediction")


# Add predict button
if st.button("Predict"):
    # Call the predict function and display the result
    prediction = predict(input_pred_rf)
    st.write(prediction)