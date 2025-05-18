import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# === Cargar modelos y utilidades ===
modelos = {
    "Random Forest": load("modelos/modelo_rf.joblib"),
    "SVM": load("modelos/modelo_svm.joblib"),
    "Gradient Boosting": load("modelos/modelo_gb.joblib"),
    "Logistic Regression": load("modelos/modelo_lr.joblib")
}
scaler = load("modelos/scaler.joblib")
selected_columns = load("modelos/columns.joblib")

# === Diccionarios de mapeo a valores numéricos ===
map_p29 = {"Conductor": 1, "Acompañante o pasajero": 2, "Peatón": 3}
map_p31 = {
    "De uso particular": 1,
    "De servicio público": 2,
    "De servicio especial": 3,
    "De uso informal": 4,
    "De uso oficial": 5
}
map_p32 = {"Sí": 1, "No": 2}
map_p35 = {
    "6:01 a.m. - 9:00 a.m.": 1,
    "9:01 a.m. - 12:00 p.m.": 2,
    "12:01 p.m. - 3:00 p.m.": 3,
    "3:01 p.m. - 6:00 p.m.": 4,
    "6:01 p.m. - 9:00 p.m.": 5,
    "9:01 p.m. - 12:00 a.m.": 6,
    "12:01 a.m. - 3:00 a.m.": 7,
    "3:01 a.m. - 6:00 a.m.": 8,
    "No recuerda": 9
}

# === Streamlit UI ===
st.title("🧠 Predicción supervisada de accidente de trafico")
modelo_nombre = st.selectbox("Selecciona el modelo de clasificación:", list(modelos.keys()))
modelo = modelos[modelo_nombre]

st.subheader("📋 Responde las siguientes preguntas:")

p29 = st.selectbox("¿Al momento de su último accidente de tránsito, usted era?", list(map_p29.keys()))
p31 = st.selectbox("¿Este vehículo era?", list(map_p31.keys()))
p32 = st.selectbox("¿Estaba realizando actividades relacionadas con su trabajo?", list(map_p32.keys()))
p35 = st.selectbox("¿Su último accidente ocurrió entre:", list(map_p35.keys()))

# === Crear DataFrame con columnas del modelo e inicializar en 0 ===
input_data = pd.DataFrame(columns=selected_columns)
input_data.loc[0] = 0

# === Asignar los valores numéricos a las columnas correspondientes ===
input_data.at[0, "p29"] = map_p29[p29]
input_data.at[0, "p31"] = map_p31[p31]
input_data.at[0, "p32"] = map_p32[p32]
input_data.at[0, "p35"] = map_p35[p35]

# === Escalar los datos ===
input_scaled = scaler.transform(input_data)

# === Botón para predecir ===
if st.button("🔍 Predecir"):
    pred = modelo.predict(input_scaled)[0]
    proba = modelo.predict_proba(input_scaled)[0][1] if hasattr(modelo, "predict_proba") else None

    if proba is not None:
        st.info(f"📊 Probabilidad de accidente: **{proba:.2%}**")
