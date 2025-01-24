import streamlit as st
import pandas as pd
import requests
import xgboost as xgb
import numpy as np

# Configuración de Supabase
SUPABASE_URL = "https://aispdrqeugwxfhghzkcd.supabase.co"  # Cambia con tu URL de Supabase
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFpc3BkcnFldWd3eGZoZ2h6a2NkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM2NzkxMzgsImV4cCI6MjA0OTI1NTEzOH0.irvfK6Wdo_OMqU29Bhz941t6-y-Zg-YuIpqXNbM3COU"  # Cambia con tu clave API
STORAGE_BUCKET = "modelos"  # Nombre del bucket
MODEL_FILENAME = "xgboost_best_model.json"  # Nombre del modelo en el bucket

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

# Función para cargar el modelo desde Supabase
def load_model_from_supabase():
    url = f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{MODEL_FILENAME}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open("temp_model.json", "wb") as f:
            f.write(response.content)
        model = xgb.Booster()
        model.load_model("temp_model.json")
        return model
    else:
        st.error(f"No se pudo cargar el modelo desde Supabase. Error: {response.text}")
        return None

# Función para realizar una consulta en Supabase
def fetch_pedidos():
    url = f"{SUPABASE_URL}/rest/v1/nuevos_registros?select=*"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error(f"Error al obtener los pedidos: {response.text}")
        return pd.DataFrame()

# Función para actualizar un pedido en Supabase
def update_pedido(pedido_id, data):
    url = f"{SUPABASE_URL}/rest/v1/nuevos_registros?id=eq.{pedido_id}"
    response = requests.patch(url, json=data, headers=headers)
    if response.status_code == 204:
        st.success("Pedido actualizado exitosamente.")
    else:
        st.error(f"Error al actualizar el pedido: {response.text}")

# Función para eliminar un pedido en Supabase
def delete_pedido(pedido_id):
    url = f"{SUPABASE_URL}/rest/v1/nuevos_registros?id=eq.{pedido_id}"
    response = requests.delete(url, headers=headers)
    if response.status_code == 204:
        st.success("Pedido eliminado exitosamente.")
    else:
        st.error(f"Error al eliminar el pedido: {response.text}")

# Función para insertar un nuevo pedido en Supabase
def insert_data_to_supabase(data):
    url = f"{SUPABASE_URL}/rest/v1/nuevos_registros"
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 201:
        st.success("Pedido creado exitosamente.")
    else:
        st.error(f"Error al crear el pedido: {response.text}")

# Función para realizar la predicción
def predict(model, record):
    # Crear DMatrix y predecir
    dmatrix = xgb.DMatrix(record)
    prediccion = model.predict(dmatrix)[0]
    return float(prediccion)

# Configuración de Streamlit
st.title("Gestión de Pedidos con Predicción")
model = load_model_from_supabase()

if model:
    # Leer pedidos existentes
    st.header("Pedidos Existentes")
    pedidos = fetch_pedidos()

    if not pedidos.empty:
        st.dataframe(pedidos)

        # Selección de pedido para modificar o eliminar
        pedido_id = st.selectbox("Selecciona un ID de pedido para modificar o eliminar", pedidos["id"].tolist())
        selected_pedido = pedidos[pedidos["id"] == pedido_id].iloc[0].to_dict()

        # Formulario para modificar el pedido
        st.subheader("Modificar Pedido")
        distancia_km = st.number_input("Distancia (km)", value=selected_pedido["distancia_km"], min_value=0.0, step=0.1)
        tiempo_preparacion_min = st.number_input("Tiempo de preparación (min)", value=selected_pedido["tiempo_preparacion_min"], min_value=0, step=1)
        experiencia_repartidor_anos = st.number_input("Experiencia del repartidor (años)", value=selected_pedido["experiencia_repartidor_anos"], min_value=0.0, step=0.1)
        clima = st.selectbox("Clima", ["Despejado", "Lluvioso", "Ventoso", "Niebla"], index=["Despejado", "Lluvioso", "Ventoso", "Niebla"].index(selected_pedido["clima"]))
        nivel_trafico = st.selectbox("Nivel de tráfico", ["Bajo", "Medio", "Alto"], index=["Bajo", "Medio", "Alto"].index(selected_pedido["nivel_trafico"]))
        momento_del_dia = st.selectbox("Momento del día", ["Mañana", "Tarde", "Noche", "Madrugada"], index=["Mañana", "Tarde", "Noche", "Madrugada"].index(selected_pedido["momento_del_dia"]))
        tipo_vehiculo = st.selectbox("Tipo de vehículo", ["Bicicleta", "Patineta", "Moto", "Auto"], index=["Bicicleta", "Patineta", "Moto", "Auto"].index(selected_pedido["tipo_vehiculo"]))

        if st.button("Actualizar Pedido"):
            # Preparar el registro para la predicción
            input_data = {
                "Distancia_km": distancia_km,
                "Tiempo_Preparacion_min": tiempo_preparacion_min,
                "Experiencia_Repartidor_anos": experiencia_repartidor_anos,
                "Clima": clima,
                "Nivel_Trafico": nivel_trafico,
                "Momento_Del_Dia": momento_del_dia,
                "Tipo_Vehiculo": tipo_vehiculo,
            }

            input_df = pd.DataFrame([input_data])
            encoded_df = pd.get_dummies(input_df, columns=["Clima", "Nivel_Trafico", "Momento_Del_Dia", "Tipo_Vehiculo"])
            expected_columns = model.feature_names
            for col in expected_columns:
                if col not in encoded_df.columns:
                    encoded_df[col] = 0
            encoded_df = encoded_df[expected_columns]

            tiempo_predicho = predict(model, encoded_df)

            # Actualizar el pedido en Supabase
            data_to_update = {
                "distancia_km": float(distancia_km),
                "tiempo_preparacion_min": int(tiempo_preparacion_min),
                "experiencia_repartidor_anos": float(experiencia_repartidor_anos),
                "clima": clima,
                "nivel_trafico": nivel_trafico,
                "momento_del_dia": momento_del_dia,
                "tipo_vehiculo": tipo_vehiculo,
                "tiempo_entrega_min": tiempo_predicho
            }
            update_pedido(pedido_id, data_to_update)

        if st.button("Eliminar Pedido"):
            delete_pedido(pedido_id)

    # Crear un nuevo pedido
    st.header("Crear Nuevo Pedido")
    with st.form("Nuevo Pedido"):
        distancia_km = st.number_input("Distancia (km)", min_value=0.0, step=0.1)
        tiempo_preparacion_min = st.number_input("Tiempo de preparación (min)", min_value=0, step=1)
        experiencia_repartidor_anos = st.number_input("Experiencia del repartidor (años)", min_value=0.0, step=0.1)
        clima = st.selectbox("Clima", ["Despejado", "Lluvioso", "Ventoso", "Niebla"])
        nivel_trafico = st.selectbox("Nivel de tráfico", ["Bajo", "Medio", "Alto"])
        momento_del_dia = st.selectbox("Momento del día", ["Mañana", "Tarde", "Noche", "Madrugada"])
        tipo_vehiculo = st.selectbox("Tipo de vehículo", ["Bicicleta", "Patineta", "Moto", "Auto"])
        submit_new = st.form_submit_button("Crear Pedido")

        if submit_new:
            # Preparar el registro para la predicción
            input_data = {
                "Distancia_km": distancia_km,
                "Tiempo_Preparacion_min": tiempo_preparacion_min,
                "Experiencia_Repartidor_anos": experiencia_repartidor_anos,
                "Clima": clima,
                "Nivel_Trafico": nivel_trafico,
                "Momento_Del_Dia": momento_del_dia,
                "Tipo_Vehiculo": tipo_vehiculo,
            }

            input_df = pd.DataFrame([input_data])
            encoded_df = pd.get_dummies(input_df, columns=["Clima", "Nivel_Trafico", "Momento_Del_Dia", "Tipo_Vehiculo"])
            expected_columns = model.feature_names
            for col in expected_columns:
                if col not in encoded_df.columns:
                    encoded_df[col] = 0
            encoded_df = encoded_df[expected_columns]

            tiempo_predicho = predict(model, encoded_df)

            # Crear el pedido en Supabase
            registro = {
                "distancia_km": float(distancia_km),
                "tiempo_preparacion_min": int(tiempo_preparacion_min),
                "experiencia_repartidor_anos": float(experiencia_repartidor_anos),
                "clima": clima,
                "nivel_trafico": nivel_trafico,
                "momento_del_dia": momento_del_dia,
                "tipo_vehiculo": tipo_vehiculo,
                "tiempo_entrega_min": tiempo_predicho
            }
            insert_data_to_supabase(registro)
