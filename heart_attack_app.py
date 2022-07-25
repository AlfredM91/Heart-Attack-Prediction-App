# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:34:51 2022

@author: aaron
"""

# Open Anaconda prompt and activate tf_env
# type cd "directory path of app.py file"
# type streamlit run app.py

import streamlit as st
import subprocess
import sys

@st.cache
def install(package):
    subprocess.check_call([sys.executable,'-m','pip','install',package])

install('pickle-mixin')
install('sklearn')

# from heart_attack_module import ModelEvaluation
# import pandas as pd 
import numpy as np
import pickle
import os

#%% Contants

MODEL_PATH = os.path.join(os.getcwd(),'models','model.pkl')
TEST_CASE = os.path.join(os.getcwd(),'dataset','test_case.csv')

#%% Model Loading

with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)

#%% Test Case Accuracy

# df_test = pd.read_csv(TEST_CASE)

# X_test = df_test.drop('output',1)
# y_test = df_test['output']

# y_pred = model.predict(X_test)

# df_test['predicted_output'] = y_pred

# me = ModelEvaluation()  
# print(me.classification_report(X_test, y_test, model, 'ml'))

#%% Streamlit

st.title('Heart Attack Prediction App')
st.write('The data for the following example is collected from the: \n\n \
1. Hungarian Institute of Cardiology. Budapest \n \
2. University Hospital, Zurich, Switzerland \n \
3. University Hospital, Basel, Switzerland \n \
4. Veteran Affairs Medical Center, Long Beach and Cleveland Clinic Foundation \n\n \
and contains information on subjects of at least 29 years old. \n\n\
This is a sample application and cannot be used as a substitute for real medical \
advice')

st.write('Please fill in the details of the person under consideration in the \
left sidebar and click on the button below')

# Forms

with st.form("Patient's info"):
    age = st.sidebar.number_input('Age in Years', 1, 150, 29, 1)
    sex = st.sidebar.radio('Sex', ('Male','Female'))
    cp = st.sidebar.selectbox('Chest Pain Type',('Typical Angina','Atypical Angina',
                              'Non-anginal Pain','Asymptomatic'))
    trtbps = st.sidebar.slider('Resting Blood Pressure', 0, 370, 120, 1)
    chol = st.sidebar.slider('Cholesterol in mg/dl', 0, 300, 60, 1)
    fbs = st.sidebar.radio('Fasting Blood Sugar', ('More than 120 mg/dl',
                            'Less than or equal to 120 mg/dl'))
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results',
                                   ('Normal',
                                    'Having ST-T wave abnormality (T wave \
                                    inversions and/or ST elevation or \
                                    depression of > 0.05 mV)',
                                    'Showing probable or definite left ventricular\
                                    hypertrophy by Estes'' criteria'))
    thalachh = st.sidebar.slider('Maximum Heart Rate',0,480,60,1)
    exng = st.sidebar.radio('Exercise Induced Angina',('Yes','No'))
    oldpeak = st.sidebar.slider('ST depression induced by exercise relative to rest',
                                0.0,10.0,1.0,0.1)
    slp = st.sidebar.selectbox('The slope of the peak exercise ST segment',
                               ('Unslopping','Flat','Downslopping'))
    caa = st.sidebar.slider('Number of major vessels',0,3,2,1)
    thall = st.sidebar.selectbox('Thalassemia',('Fixed defect','Normal',
                                                'Reversable Defect'))
    
    if sex == 'Male':
        sex = 1
    else:
        sex = 0
    
    if cp == 'Typical Angina':
        cp = 0
    elif cp == 'Atypical Angina':
        cp = 1
    elif cp == 'Non-anginal Pain':
        cp = 2
    else:
        cp = 3
    
    if fbs == 'More than 120 mg/dl':
        fbs = 1
    else:
        fbs = 0
    
    if restecg == 'Normal':
        restecg = 0
    elif restecg == 'Having ST-T wave abnormality (T wave inversions and/or \
        ST elevation or depression of > 0.05 mV)':
        restecg = 1
    else:
        restecg = 2
        
    if exng == 'Yes':
        exng = 1
    else:
        exng = 0
        
    if slp == 'Unslopping':
        slp = 0
    elif slp == 'Flat':
        slp = 1
    else:
        slp = 2
        
    if thall == 'Fixed defect':
        thall = 1
    elif thall == 'Normal':
        thall = 2
    else:
        thall = 3
    
    row = [age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]
    
    submitted = st.form_submit_button('Submit')
    
    if submitted:
        new_data = np.expand_dims(row,axis=0)
        outcome = model.predict(new_data)[0]
        
        if outcome==0:
            st.write('Congrats! You are not at risk of heart attack! Keep \
                     up with the healthy lifestyle :)')
            st.balloons()
        else:
            st.warning('You are at high risk of heart attack. Please consult \
                     a medical professional for preventative measures')
            
    