import streamlit as st
import pandas as pd
import requests
import joblib
import io
import numpy as np

# Configuración de Supabase
SUPABASE_URL = "https://aispdrqeugwxfhghzkcd.supabase.co"  # Cambia con tu URL de Supabase
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFpc3BkcnFldWd3eGZoZ2h6a2NkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM2NzkxMzgsImV4cCI6MjA0OTI1NTEzOH0.irvfK6Wdo_OMqU29Bhz941t6-y-Zg-YuIpqXNbM3COU"  # Cambia con tu clave API
STORAGE_BUCKET = "modelorf"  # Nombre del bucket
MODEL_FILENAME = "best_random_forest_model.pkl"  # Nombre del modelo en el bucket

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

# Función para cargar el modelo desde el almacenamiento de Supabase
def load_model_from_supabase():
    url = f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{MODEL_FILENAME}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        model = joblib.load(io.BytesIO(response.content))
        return model
    else:
        st.error(f"No se pudo cargar el modelo desde Supabase. Error: {response.text}")
        return None

# Función para insertar un nuevo registro en Supabase
def insert_data_to_supabase(data):
    url = f"{SUPABASE_URL}/rest/v1/nuevos_registros"
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 201:
        st.success("Registro guardado exitosamente en Supabase.")
    else:
        st.error(f"Error al guardar el registro: {response.text}")

# Configuración de Streamlit
st.title("Predicción de Tiempos de Entrega")
st.write("Introduce los datos del pedido para calcular el tiempo estimado de entrega.")

# Formulario para ingresar datos
with st.form("Formulario de Predicción"):
    distancia_km = st.number_input("Distancia (km)", min_value=0.0, step=0.1)
    tiempo_preparacion_min = st.number_input("Tiempo de preparación (min)", min_value=0, step=1)
    experiencia_repartidor_anos = st.number_input("Experiencia del repartidor (años)", min_value=0.0, step=0.1)
    clima = st.selectbox("Clima", ["Despejado", "Lluvioso", "Ventoso", "Niebla"])
    nivel_trafico = st.selectbox("Nivel de tráfico", ["Bajo", "Medio", "Alto"])
    momento_del_dia = st.selectbox("Momento del día", ["Mañana", "Tarde", "Noche", "Madrugada"])
    tipo_vehiculo = st.selectbox("Tipo de vehículo", ["Bicicleta", "Patineta", "Moto", "Auto"])

    submit = st.form_submit_button("Calcular Predicción")

# Procesar datos al enviar el formulario
if submit:
    # Cargar el modelo desde Supabase
    model = load_model_from_supabase()

    if model:
        # Preprocesar el registro (modificar según las necesidades del modelo)
        try:
            # Ajustar si el modelo requiere procesamiento adicional (e.g., escalado, encoding)
            record = np.array([[distancia_km, tiempo_preparacion_min, experiencia_repartidor_anos]])  # Ajustar si es necesario

            # Realizar la predicción
            tiempo_predicho = model.predict(record)[0]

            # Mostrar el resultado
            st.write(f"**Tiempo estimado de entrega:** {tiempo_predicho:.2f} minutos")

            # Preparar el registro para la base de datos
            registro = {
                "distancia_km": distancia_km,
                "tiempo_preparacion_min": tiempo_preparacion_min,
                "experiencia_repartidor_anos": experiencia_repartidor_anos,
                "clima": clima,
                "nivel_trafico": nivel_trafico,
                "momento_del_dia": momento_del_dia,
                "tipo_vehiculo": tipo_vehiculo,
                "tiempo_entrega_min": tiempo_predicho
            }

            # Guardar el registro en Supabase
            insert_data_to_supabase(registro)

        except Exception as e:
            st.error(f"Error durante la predicción: {e}")
