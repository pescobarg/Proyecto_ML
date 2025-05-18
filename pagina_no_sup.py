import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# === Cargar modelos y utilidades ===
modelos_ns = {
    "KMeans": load("modelos/modelo_KM.joblib"),
    "Gaussian Mixture": load("modelos/modelo_GMM.joblib"),
    "DBSCAN": load("modelos/modelo_DB.joblib")
}
scaler = load("modelos/scaler_NS.joblib")
selected_columns = load("modelos/columns_NS.joblib")
pca = load("modelos/pca_NS.joblib")  # Si guardaste el PCA también

# Map para mostrar clusters con clases (opcional)
# Aquí puedes ajustar según tu lógica de asignación mayoritaria, o mostrar cluster directo
cluster_to_class = {0: "Clase 0", 1: "Clase 1"}  # Ejemplo simple, adapta según tu caso

st.title("🧠 Clasificación no supervisada de accidente de tráfico")

modelo_nombre = st.selectbox("Selecciona el modelo no supervisado:", list(modelos_ns.keys()))
modelo = modelos_ns[modelo_nombre]

st.subheader("📋 Ingresa las características:")

# Mismo mapeo que supervisado para inputs
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

# === Inputs ===
p29 = st.selectbox("¿Al momento de su último accidente de tránsito, usted era?", list(map_p29.keys()))
p31 = st.selectbox("¿Este vehículo era?", list(map_p31.keys()))
p32 = st.selectbox("¿Estaba realizando actividades relacionadas con su trabajo?", list(map_p32.keys()))
p35 = st.selectbox("¿Su último accidente ocurrió entre:", list(map_p35.keys()))

# === Crear DataFrame con columnas del modelo e inicializar en 0 ===
input_data = pd.DataFrame(columns=selected_columns)
input_data.loc[0] = 0

# Asignar valores
input_data.at[0, "p29"] = map_p29[p29]
input_data.at[0, "p31"] = map_p31[p31]
input_data.at[0, "p32"] = map_p32[p32]
input_data.at[0, "p35"] = map_p35[p35]

# Escalar y aplicar PCA
input_scaled = scaler.transform(input_data)
input_pca = pca.transform(input_scaled)

# Predecir y mostrar resultado
if st.button("🔍 Predecir cluster"):

    if modelo_nombre == "DBSCAN":
        st.warning("DBSCAN no permite predecir nuevos datos directamente. Usa KMeans o GMM para predicción.")
    else:
        clusters = modelo.predict(input_pca)
        cluster = clusters[0]
        st.success(f"Cluster asignado: {cluster}")
        if modelo_nombre == "Gaussian Mixture":
            proba = modelo.predict_proba(input_pca)[0][cluster]
            st.info(f"Probabilidad de pertenencia al cluster: {proba:.2%}")
        elif modelo_nombre == "KMeans":
            centros = modelo.cluster_centers_
            distancia = np.linalg.norm(input_pca - centros[cluster])
            confianza = 1 / (1 + distancia)
            st.info(f"Confianza estimada (proxy): {confianza:.2%}")
