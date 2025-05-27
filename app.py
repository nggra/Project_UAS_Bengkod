import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# Load dataset
file_path = "ObesityDataSet.csv"
df = pd.read_csv(file_path)

# Konversi kolom numerik ke float
numeric_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Bersihkan data
df_clean = df.copy()
df_clean = df_clean.drop_duplicates()

# Imputasi kategorikal
cat_columns = df_clean.select_dtypes(include='object').columns.drop('NObeyesdad')
cat_imputer = SimpleImputer(strategy='most_frequent')
df_clean[cat_columns] = cat_imputer.fit_transform(df_clean[cat_columns])

# Hapus outlier
for col in numeric_columns:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

# Encode
encoder = LabelEncoder()
for col in cat_columns:
    df_clean[col] = encoder.fit_transform(df_clean[col])

target_encoder = LabelEncoder()
df_clean['NObeyesdad'] = target_encoder.fit_transform(df_clean['NObeyesdad'])

# Normalisasi
scaler = StandardScaler()
df_clean[numeric_columns] = scaler.fit_transform(df_clean[numeric_columns])

# Simpan scaler
joblib.dump(scaler, "scaler.pkl")

# Pisah fitur dan target
X = df_clean.drop('NObeyesdad', axis=1)
y = df_clean['NObeyesdad']

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['NObeyesdad'] = y_resampled
df_resampled['NObeyesdad_Label'] = target_encoder.inverse_transform(y_resampled)
df_resampled.to_csv("data_after_smote.csv", index=False)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=0, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Simpan model terbaik
joblib.dump(best_rf, "random_forest_model.pkl")

# Aplikasi Streamlit
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Tingkat Obesitas")
st.write("Masukkan informasi berikut untuk memprediksi status berat badan Anda:")

age = st.slider("Usia", 10, 100)
height = st.slider("Tinggi Badan (meter)", 1.0, 2.5, step=0.01)
weight = st.slider("Berat Badan (kg)", 20.0, 200.0, step=0.5)
fcvc = st.slider("Konsumsi Sayur (1 - jarang, 3 - sering)", 1, 3)
ncp = st.slider("Jumlah makan besar per hari", 1, 4)
ch2o = st.slider("Konsumsi air harian (1 - sedikit, 3 - banyak)", 1, 3)
faf = st.slider("Frekuensi aktivitas fisik (0 - tidak pernah, 3 - rutin)", 0, 3)
tue = st.slider("Waktu screen time (jam/hari)", 0, 3)

input_data = np.array([[age, height, weight, fcvc, ncp, ch2o, faf, tue]])
input_scaled = scaler.transform(input_data)

label_map = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}

if st.button("Prediksi"):
    prediction = model.predict(input_scaled)
    label = label_map.get(prediction[0], "Unknown")
    st.success(f"Hasil Prediksi: {label}")
