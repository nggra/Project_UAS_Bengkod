import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Muat model dan metadata
try:
    data = joblib.load("random_forest_model.pkl")
    model = data['model']
    expected_columns = data['columns']
    scaler = data['scaler']
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan file `random_forest_model.pkl` sudah diupload.")
    st.stop()

# Mapping hasil prediksi
label_map = {
    0: "Insufficient_Weight", 
    1: "Normal_Weight", 
    2: "Overweight_Level_I",
    3: "Overweight_Level_II", 
    4: "Obesity_Type_I",
    5: "Obesity_Type_II", 
    6: "Obesity_Type_III"
}

# UI Streamlit
st.title("🎯 Prediksi Tingkat Obesitas")
st.subheader("Isi data berikut:")

# Input user
age = st.slider("Usia", 10, 100)
height = st.slider("Tinggi Badan (meter)", 1.0, 2.5, step=0.01)
weight = st.slider("Berat Badan (kg)", 20.0, 200.0, step=0.5)
fcvc = st.slider("Konsumsi Sayur (1 - jarang, 3 - sering)", 1, 3)
ncp = st.slider("Jumlah makan besar per hari", 1, 4)
ch2o = st.slider("Konsumsi air harian (1 - sedikit, 3 - banyak)", 1, 3)
faf = st.slider("Frekuensi aktivitas fisik (0 - tidak pernah, 3 - rutin)", 0, 3)
tue = st.slider("Waktu screen time (jam/hari)", 0, 3)

gender = st.selectbox("Gender", options=["Male", "Female"])
family_history = st.selectbox("Riwayat Keluarga Obesitas", options=["no", "yes"])
favc = st.selectbox("Frequent consumption of high-calorie food", options=["no", "yes"])
caec = st.selectbox("Consumption of food between meals", options=["Sometimes", "Frequently", "Always", "no"])
smoke = st.selectbox("Smoking", options=["no", "yes"])
scc = st.selectbox("Calories consumption monitoring", options=["no", "yes"])
calc = st.selectbox("Alcohol consumption", options=["no", "Sometimes", "Frequently"])
mtrans = st.selectbox("Transportation used", options=["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# Buat DataFrame
input_dict = {
    'Age': [age],
    'Gender': [gender],
    'Height': [height],
    'Weight': [weight],
    'FamilyHistory': [family_history],
    'FAVC': [favc],
    'FCVC': [fcvc],
    'NCP': [ncp],
    'CAEC': [caec],
    'SMOKE': [smoke],
    'CH2O': [ch2o],
    'SCC': [scc],
    'FAF': [faf],
    'TUE': [tue],
    'CALC': [calc],
    'MTRANS': [mtrans]
}

input
