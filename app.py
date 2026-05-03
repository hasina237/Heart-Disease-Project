import streamlit as st 
import pandas as pd 
import joblib

model = joblib.load('heart_model.pkl')
st.title('Heart Disease prediction App')

st.write('Enter pationt details ')

# inputs
age = st.number_input('age',1,120,50)
gender = st.selectbox('gender (1= male, 0= female)',[1,0])
height = st.number_input('height (cm)',value=170)
weight = st.number_input('weight (kg)', value=70)
ap_hi = st.number_input('ap_hi',value=80)
ap_lo = st.number_input('ap_lo',value=80)
cholesterol = st.selectbox('cholesterol (1-3)',[1,2,3])
gluc = st.selectbox('glucose (1-3)',[1,2,3])
smoke = st.selectbox('smoke', [0,1])
alco = st.selectbox('alcohol',[0,1])
active = st.selectbox('active',[0,1])

# dataframe
input_data = pd.DataFrame({
    'age': [age],
    'gender':[gender],
    'height':[height],
    'weight':[weight],
    'ap_hi': [ap_hi],
    'ap_lo':[ap_lo],
    'cholesterol':[cholesterol],
    'gluc':[gluc],
    'smoke':[smoke],
    'alco':[alco],
    'active':[active]
})

# predict

if st.button('Predict'):
    result = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if result == 1:
        st.error(f"High Risk :{prob * 100 : 2f} %")
    else:
        st.success(f"Low Risk :{(1- prob)*100: 2f}")