import streamlit as st
import pandas as pd
import numpy as np
import joblib
import keras
from scripts.ml_script import KFoldTargetEncoder
from sklearn.ensemble import RandomForestRegressor

st.title('Used Car Price Predictor')

st.markdown('##### Goal: Compare ML and DL models to predict used car prices based on real-world vehicle data.')

st.caption('Fill out the following information to get your prediction. Leave blank if unknown.')
# input_data = None
# data_entered = False
data_entered = True
input_data = pd.DataFrame([{
    "milage": 7008.0,
    "accident": "Unknown",
    "clean_title": "Missing",
    "brand_model": "Porsche 911 Carrera S",
    "car_age": 3,
    "engine_hp": 443.0,
    "engine_liter": 3.0,
    "engine_cyl": 6.0,
    "fuel_type": "Gasoline",
    "trans_type": "auto",
    "trans_spd": 8.0
}])
with st.expander("Your Car Info Goes Here"):
    brand = (st.text_input("Car Brand")) or None
    model = (st.text_input("Car Model")) or None
    brand_model = None if (brand == None or model == None) else brand + " " + model

    accident = st.selectbox("Accidents", ["Unknown", "None reported", "At least 1 accident or damage reported"])

    clean_title = st.selectbox("Clean Title", ["Yes", "No"])
    clean_title = 'Missing' if clean_title == 'No' else clean_title

    milage = st.number_input("Mileage", step=1)
    
    age = st.number_input("Car Age (Years)", step=1)

    hp = st.number_input("Engine Horse Power", step=1)
    engine_liter = st.number_input("Engine Total Volume (L)", step=1)
    engine_cyl = st.number_input("Engine Number of Cylinders", step=1)

    fuel_type = st.selectbox("Fuel Type", ["–", "Gasoline", "Hybrid", "Electric", "Diesel", "Plug-In Hybrid", "E85 Flex Fuel"])

    trans_type = st.selectbox("Transmission Type", ["–", "auto", "manual", "dual"])
    trans_type = 'unknown' if trans_type == '–' else trans_type
    trans_spd = st.number_input("Number of Gears", step=1)


    if st.button('Submit'):
        input_data = {
            "brand_model": [brand_model],
            "accident": [accident],
            "clean_title": [clean_title],
            "milage": [milage],
            "car_age": [age],
            "engine_hp": [hp],
            "engine_liter": [engine_liter],
            "engine_cyl": [engine_cyl],
            "fuel_type": [fuel_type],
            "trans_spd": [trans_spd],
            "trans_type": [trans_type]
        }
        data_entered = True


m = 'dl'

with st.sidebar:
    st.header('Toggle Model')
    if st.button('ML XGB Model'):
        m = 'ml'
    if st.button('DL Keras Model'):
        m = 'dl'

st.divider()

# No Model Selected
if m == 'a':
    st.markdown('##### Select a model in the sidebar to get started!')



# ML Model
if m == 'ml':
    st.header('Machine Learning: XGB Model')
    if data_entered:
        st.subheader("Inputted data sample")
        st.write(input_data)
        pipe = joblib.load('./models/ml_model.joblib')
        pred = pipe.predict(input_data)
        res = np.expm1(pred[0])
        st.subheader("Prediction:")
        st.success(f"Estimated price (as of 01/2025): **${res:,.0f}**")




# DL Model
if m == 'dl':
    st.header('Deep Learning: Keras Model')
    if data_entered:
        st.subheader("Inputted data sample")
        st.write(input_data)
        model = keras.models.load_model("./models/dl_model.keras")
        preprocessor = joblib.load("./models/dl_preprocessor.joblib")
        input_prep = preprocessor.transform(input_data)
        pred = model.predict(input_prep) 
        # res = np.expm1(pred[0])
        st.subheader("Prediction:")
        st.success(f"Estimated price (as of 01/2025): **${pred[0][0]:,.0f}**")