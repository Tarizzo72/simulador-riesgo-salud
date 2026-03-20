import streamlit as st
import joblib
import pandas as pd

# ===============================
# CONFIGURACIÓN
# ===============================

st.set_page_config(page_title="Riesgo de Salud", layout="wide")

# Cargar modelo
model = joblib.load('modelo.pkl')
columns = joblib.load('columnas.pkl')

st.title("🧠 Simulador de Riesgo de Salud en Estudiantes")
st.markdown("Ajustá los valores y observá cómo cambia el riesgo en tiempo real.")

# ===============================
# LAYOUT EN COLUMNAS
# ===============================

col1, col2 = st.columns(2)

# ===============================
# INPUTS
# ===============================

with col1:
    st.subheader("🧠 Estrés")

    stress_self = st.slider("Estrés percibido", 0.0, 10.0, 5.0)
    stress_bio = st.slider("Estrés fisiológico", 0.0, 10.0, 5.0)

    st.subheader("📚 Actividad académica")

    study_hours = st.slider("Horas de estudio", 0, 12, 5)
    project_hours = st.slider("Horas de proyectos", 0, 12, 5)

with col2:
    st.subheader("❤️ Salud física")

    heart_rate = st.slider("Frecuencia cardíaca", 50, 120, 70)
    systolic = st.slider("Presión sistólica", 90, 140, 120)
    diastolic = st.slider("Presión diastólica", 60, 100, 80)

    st.subheader("🌿 Hábitos")

    sleep_quality = st.selectbox("Calidad del sueño", ["Good", "Poor"])
    physical_activity = st.selectbox("Actividad física", ["Low", "Moderate", "High"])

# ===============================
# PREPARAR INPUT
# ===============================

input_dict = {
    'Stress_Level_Self_Report': stress_self,
    'Stress_Level_Biosensor': stress_bio,
    'Heart_Rate': heart_rate,
    'Blood_Pressure_Systolic': systolic,
    'Blood_Pressure_Diastolic': diastolic,
    'Study_Hours': study_hours,
    'Project_Hours': project_hours,
    'Sleep_Quality_Poor': 1 if sleep_quality == "Poor" else 0,
    'Physical_Activity_Moderate': 1 if physical_activity == "Moderate" else 0,
    'Physical_Activity_High': 1 if physical_activity == "High" else 0
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=columns, fill_value=0)

# ===============================
# PREDICCIÓN
# ===============================

prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0]
classes = model.classes_

# ===============================
# OUTPUT
# ===============================

st.markdown("---")
st.subheader("📊 Resultado del análisis")

col3, col4 = st.columns(2)

# 🔥 Resultado visual
with col3:
    if prediction == "Low":
        st.success("🟢 Bajo riesgo")
    elif prediction == "Moderate":
        st.warning("🟡 Riesgo moderado")
    else:
        st.error("🔴 Alto riesgo")

# 🔢 Probabilidades
with col4:
    st.write("Probabilidad por clase:")
    for i, clase in enumerate(classes):
        st.write(f"{clase}: {proba[i]*100:.2f}%")

# ===============================
# INTERPRETACIÓN
# ===============================

st.markdown("---")
st.subheader("🧠 Interpretación del modelo")

stress_avg = (stress_self + stress_bio) / 2

if stress_avg > 7:
    st.error("El alto nivel de estrés es el principal factor de riesgo.")

elif stress_avg < 3:
    st.success("El bajo nivel de estrés reduce significativamente el riesgo.")

if study_hours > 8 or project_hours > 8:
    st.info("Alta carga académica puede incrementar el estrés.")

if heart_rate > 90:
    st.info("Frecuencia cardíaca elevada puede indicar impacto fisiológico.")