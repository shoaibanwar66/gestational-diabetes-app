import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image


# Load  model a 
model = joblib.load(open("model-v1.joblib","rb"))

def data_preprocessor(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    return df

def visualize_confidence_level(prediction_proba, processed_user_input):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    
    predict_diabetics = (prediction_proba[0][1]*100).round(2)
    print (predict_diabetics)
    st.metric(label="Chances of Diabetics", value= str(predict_diabetics) + " %")
    age = processed_user_input._get_value(0, 'Age' )
    BMI = processed_user_input._get_value(0, 'BMI' )
    if age< 18 :
        age_group = 'Young'
    elif age >= 19 & age < 55:
        age_group = 'Mature'
    elif (age >= 56) :
        age_group = 'Old'
        
        
    if  (age < 34):
        age_vs_diabete = 'Low Risky'
    elif (age>= 34) & (age < 44):
        age_vs_diabete = 'Risky'
    elif (age >= 45) & (age < 55):
        age_vs_diabete = 'Too Risky'
    elif (age >= 56):
        age_vs_diabete = "High Risk"
        
    if BMI< 18:
        weight_criteria = 'Unhealthy'
    elif (BMI>= 19) & (BMI< 25):
        weight_criteria = 'Normal'
    elif (BMI>= 26) & (BMI< 30):
        weight_criteria = "Overweight"
    elif BMI>= 31:
        weight_criteria = 'Obese'
        
    col1, col2, col3 = st.columns(3)
    col1.metric("Age Group to be pregnant", age_group)
    col2.metric("Risk level in case of diabetic", age_vs_diabete)
    col3.metric("Weight criteria", weight_criteria)
    

    return

st.write("""Diabetic prediction during pregnancies 
This app predicts the possibilility of diabetics using input features via the side panel 
This is only 78% accurate. You should consult your doctor and have regular check-up.""")

#read in wine image and render with streamlit
image = Image.open('DiabetesPregnancy.jpg')
st.image(image, caption='Diabetes during pregnancy',use_column_width=True)

st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe

    """
    
    Age = st.sidebar.slider("Your age", 15, 85, 18)
    Pregnancies = st.sidebar.slider('Number of pregnancies', 0, 40, 0)
    Glucose = st.sidebar.slider('Current glucose level', 25., 200., 140.)
    BloodPressure  = st.sidebar.slider('Current blood pressure', 15., 200., 60.)
    SkinThickness  = st.sidebar.slider('SkinThickness', 5., 60., 20.)
    Insulin  = st.sidebar.slider('Insulin level in mIU/L 2 hour after glucose admistration', 1., 500., 15.0)
    BMI = st.sidebar.slider('BMI', 12., 60., 22.)
    DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function (family history on diabetics)', 0.05, 2., 0.1)

    features = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age
            }
    data = pd.DataFrame(features,index=[0])

    return data

user_input_df = get_user_input()
processed_user_input = data_preprocessor(user_input_df)

st.subheader('User Input parameters')
st.write(user_input_df)

prediction = model.predict(processed_user_input)
prediction_proba = model.predict_proba(processed_user_input)
#print (prediction_proba, user_input_df)
visualize_confidence_level(prediction_proba, user_input_df)