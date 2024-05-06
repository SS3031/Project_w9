## This project focuses on banking data. 
### Introduction
This project explores the dataset to predict credit risk. The analysis focuses on understanding how various factors, including marital status, gender, housing status, age, past credits and job status affect credit risk.

## Data Used:
data sourced from: https://www.kaggle.com/datasets/arunjangir245/german-credit-card

Challenges:
Since the marital status and sex column were combined, It posed problems while classifying based on these two factors. The dataset is old and currency is in Mark.

## Important EDA questions:
1. Age group which gets most credits at the bank?
2. What are the purpose for getting credits?
3. Gender distribution of the credit receivers? 
4. Employmment background of the credit receivers? 


## Methodology Description:
### Data Cleaning:

# Dropping Columns: Removed unnecessary columns from the dataset to focus on relevant variables for analysis.
# Splitting marital_status_gender column
# Categorizing age into age groups:
18 to 24 ==  Young_professionals
25 to 35 ==  Middle-aged_professionals
36 to 50 == Older_professionals
51 to 64 == Seniors
65 to 76 == Retire

Handling spaces in categorical entries

## Performing EDA to answer the above mentioned EDA questions.

### Splitting the dataset into test-train set and using RandomForest Classifier for the model training and evaluation
## USing Steamlit for showing predictive outcome from the model training.