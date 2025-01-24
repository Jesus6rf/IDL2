import streamlit as st
import pandas as pd
import requests
import xgboost as xgb
import numpy as np

# Configuración de Supabase
SUPABASE_URL = "https://aispdrqeugwxfhghzkcd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFpc3BkcnFldWd3eGZoZ2h6a2NkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM2NzkxMzgsImV4cCI6MjA0OTI1NTEzOH0.irvfK6Wdo_OMqU29Bhz941t6-y-Zg-YuIpqXNbM3COU"
STORAGE_BUCKET = "modelos"
MODEL_FILENAME = "xgboost_best_model.json"

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

# Función para obtener los pedidos desde Supabase
def fetch_pedidos():
    url = f"{SUPABASE_URL}/rest/v1/nuevos_registros?select=*"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error(f"Error al obtener los pedidos: {response.text}")
        return pd.DataFrame()

# Configuración de Streamlit
st.title("Gestión de Pedidos con Predicción")
model = load_model_from_supabase()

if model:
    # Tabs para dividir las secciones
    tab1, tab2, tab3, tab4 = st.tabs(["Nuevo Pedido", "Modificar Pedido", "Borrar Pedido", "Buscar Pedido"])

    # Tab: Nuevo Pedido
    with tab1:
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
                registro = {
                    "distancia_km": float(distancia_km),
                    "tiempo_preparacion_min": int(tiempo_preparacion_min),
                    "experiencia_repartidor_anos": float(experiencia_repartidor_anos),
                    "clima": clima,
                    "nivel_trafico": nivel_trafico,
                    "momento_del_dia": momento_del_dia,
                    "tipo_vehiculo": tipo_vehiculo,
                }
                st.success("Pedido creado exitosamente.")

    # Tab: Modificar Pedido
    with tab2:
        st.header("Modificar Pedido")
        pedidos = fetch_pedidos()
        if not pedidos.empty:
            st.dataframe(pedidos)
            pedido_id = st.selectbox("Selecciona un ID para modificar", pedidos["id"].tolist())
            selected_pedido = pedidos[pedidos["id"] == pedido_id].iloc[0].to_dict()
            distancia_km = st.number_input("Distancia (km)", value=float(selected_pedido["distancia_km"]), min_value=0.0, step=0.1)
            tiempo_preparacion_min = st.number_input("Tiempo de preparación (min)", value=int(selected_pedido["tiempo_preparacion_min"]), min_value=0, step=1)
            experiencia_repartidor_anos = st.number_input("Experiencia del repartidor (años)", value=float(selected_pedido["experiencia_repartidor_anos"]), min_value=0.0, step=0.1)
            clima = st.selectbox("Clima", ["Despejado", "Lluvioso", "Ventoso", "Niebla"], index=["Despejado", "Lluvioso", "Ventoso", "Niebla"].index(selected_pedido["clima"]))
            if st.button("Actualizar Pedido"):
                st.success("Pedido actualizado correctamente.")

    # Tab: Borrar Pedido
    with tab3:
        st.header("Borrar Pedido")
        pedidos = fetch_pedidos()
        if not pedidos.empty:
            st.dataframe(pedidos)
            pedido_id = st.selectbox("Selecciona un ID para eliminar", pedidos["id"].tolist())
            if st.button("Eliminar Pedido"):
                st.success("Pedido eliminado correctamente.")

    # Tab: Buscar Pedido
    with tab4:
        st.header("Buscar Pedido")
        pedidos = fetch_pedidos()
        if not pedidos.empty:
            search_by = st.selectbox("Buscar por", ["ID", "Clima", "Nivel de Tráfico", "Momento del Día", "Tipo de Vehículo"])
            query = st.text_input("Ingresa el valor a buscar")
            if st.button("Buscar"):
                if search_by == "ID":
                    resultados = pedidos[pedidos["id"] == int(query)]
                else:
                    resultados = pedidos[pedidos[search_by.lower()] == query]
                if not resultados.empty:
                    st.dataframe(resultados)
                else:
                    st.warning("No se encontraron resultados para la búsqueda.")
