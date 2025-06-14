import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Muat model dan scaler
try:
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("File model atau scaler tidak ditemukan. Pastikan file sudah diupload.")
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
st.write("Isi data berikut untuk memprediksi tingkat obesitas Anda:")

# Input user
age = st.slider("Usia", 10, 100)
gender = st.selectbox("Jenis Kelamin", options=["Male", "Female"])
height = st.slider("Tinggi Badan (meter)", 1.0, 2.5, step=0.01)
weight = st.slider("Berat Badan (kg)", 20.0, 200.0, step=0.5)
family_history = st.selectbox("Riwayat keluarga obesitas", options=["no", "yes"])
favc = st.selectbox("Frequent consumption of high-calorie food", options=["no", "yes"])
fcvc = st.slider("Konsumsi sayur (1 - jarang, 2 - kadang-kadang, 3 - sering)", 1, 3)
ncp = st.slider("Jumlah makan besar per hari", 1, 4)
caec = st.selectbox("Consumption of food between meals", options=["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Merokok?", options=["no", "yes"])
ch2o = st.slider("Konsumsi air harian (1 - sedikit, 2 - cukup, 3 - banyak)", 1, 3)
scc = st.selectbox("Monitoring konsumsi kalori", options=["no", "yes"])
faf = st.slider("Frekuensi aktivitas fisik (0 - tidak pernah, 3 - rutin)", 0, 3)
tue = st.slider("Waktu screen time (jam/hari)", 0, 3)
calc = st.selectbox("Alcohol consumption", options=["no", "Sometimes", "Frequently"])
mtrans = st.selectbox("Transportasi yang digunakan", options=["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# Encode manual kategorikal
gender_encoded = 1 if gender == "Male" else 0
family_history_encoded = 1 if family_history == "yes" else 0
favc_encoded = 1 if favc == "yes" else 0
caec_encoded = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec]
smoke_encoded = 1 if smoke == "yes" else 0
scc_encoded = 1 if scc == "yes" else 0
calc_encoded = {"no": 0, "Sometimes": 1, "Frequently": 2}[calc]
mtrans_encoded = {"Public_Transportation": 0, "Walking": 1, "Automobile": 2, "Motorbike": 3, "Bike": 4}[mtrans]

# Gabungkan semua fitur dalam urutan yang benar (sesuaikan dengan saat training)
input_data = np.array([[
    age,
    gender_encoded,
    height,
    weight,
    family_history_encoded,
    favc_encoded,
    fcvc,
    ncp,
    caec_encoded,
    smoke_encoded,
    ch2o,
    scc_encoded,
    faf,
    tue,
    calc_encoded,
    mtrans_encoded
]])

# Standarisasi input
try:
    input_scaled = scaler.transform(input_data)
except Exception as e:
    st.error(f"Gagal melakukan scaling: {e}")
    st.stop()

# Prediksi
if st.button("🔍 Prediksi"):
    try:
        prediction = model.predict(input_scaled)
        label = label_map.get(prediction[0], "Unknown")
        st.success(f"🎯 Hasil Prediksi: **{label}**")
    except Exception as e:
        st.error(f"🚨 Gagal melakukan prediksi: {e}")
