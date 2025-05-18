import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# ====================
# Cargar modelos
# ====================
modelos_sup = {
    "Random Forest": load("modelos/modelo_rf.joblib"),
    "SVM": load("modelos/modelo_svm.joblib"),
    "Gradient Boosting": load("modelos/modelo_gb.joblib"),
    "Logistic Regression": load("modelos/modelo_lr.joblib")
}
scaler_sup = load("modelos/scaler.joblib")
selected_columns_sup = load("modelos/columns.joblib")

modelos_ns = {
    "KMeans": load("modelos/modelo_KM.joblib"),
    "Gaussian Mixture": load("modelos/modelo_GMM.joblib"),
    "DBSCAN": load("modelos/modelo_DB.joblib")
}
scaler_ns = load("modelos/scaler_NS.joblib")
selected_columns_ns = load("modelos/columns_NS.joblib")
pca = load("modelos/pca_NS.joblib")

# ====================
# Diccionarios de mapeo
# ====================
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

# ====================
# Lógica para cambiar de vista
# ====================
if "pagina" not in st.session_state:
    st.session_state.pagina = "supervisada"

col1, col2 = st.columns(2)
with col1:
    if st.button("🔁 Ir a Supervisada"):
        st.session_state.pagina = "supervisada"
with col2:
    if st.button("🔁 Ir a No Supervisada"):
        st.session_state.pagina = "no_supervisada"

# ====================
# Página Supervisada
# ====================
if st.session_state.pagina == "supervisada":
    st.title("🧠 Predicción supervisada de accidente de tráfico")
    modelo_nombre = st.selectbox("Selecciona el modelo de clasificación:", list(modelos_sup.keys()))
    modelo = modelos_sup[modelo_nombre]

    st.subheader("📋 Responde las siguientes preguntas:")
    p29 = st.selectbox("¿Al momento de su último accidente de tránsito, usted era?", list(map_p29.keys()))
    p31 = st.selectbox("¿Este vehículo era?", list(map_p31.keys()))
    p32 = st.selectbox("¿Estaba realizando actividades relacionadas con su trabajo?", list(map_p32.keys()))
    p35 = st.selectbox("¿Su último accidente ocurrió entre:", list(map_p35.keys()))

    input_data = pd.DataFrame(columns=selected_columns_sup)
    input_data.loc[0] = 0
    input_data.at[0, "p29"] = map_p29[p29]
    input_data.at[0, "p31"] = map_p31[p31]
    input_data.at[0, "p32"] = map_p32[p32]
    input_data.at[0, "p35"] = map_p35[p35]

    input_scaled = scaler_sup.transform(input_data)

    if st.button("🔍 Predecir (Supervisado)"):
        pred = modelo.predict(input_scaled)[0]
        proba = modelo.predict_proba(input_scaled)[0][1] if hasattr(modelo, "predict_proba") else None

        st.success(f"Clase predicha: {pred}")
        if proba is not None:
            st.info(f"📊 Probabilidad de accidente: **{proba:.2%}**")

# ====================
# Página No Supervisada
# ====================
elif st.session_state.pagina == "no_supervisada":
    st.title("🧠 Clasificación no supervisada de accidente de tráfico")
    modelo_nombre = st.selectbox("Selecciona el modelo no supervisado:", list(modelos_ns.keys()))
    modelo = modelos_ns[modelo_nombre]

    st.subheader("📋 Ingresa las características:")
    p29 = st.selectbox("¿Al momento de su último accidente de tránsito, usted era?", list(map_p29.keys()))
    p31 = st.selectbox("¿Este vehículo era?", list(map_p31.keys()))
    p32 = st.selectbox("¿Estaba realizando actividades relacionadas con su trabajo?", list(map_p32.keys()))
    p35 = st.selectbox("¿Su último accidente ocurrió entre:", list(map_p35.keys()))

    input_data = pd.DataFrame(columns=selected_columns_ns)
    input_data.loc[0] = 0
    input_data.at[0, "p29"] = map_p29[p29]
    input_data.at[0, "p31"] = map_p31[p31]
    input_data.at[0, "p32"] = map_p32[p32]
    input_data.at[0, "p35"] = map_p35[p35]

    input_scaled = scaler_ns.transform(input_data)
    input_pca = pca.transform(input_scaled)

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
