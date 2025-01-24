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

# Función para insertar un pedido en Supabase
def insert_data_to_supabase(data):
    url = f"{SUPABASE_URL}/rest/v1/nuevos_registros"
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 201:
        st.success("Pedido creado exitosamente.")
    else:
        st.error(f"Error al crear el pedido: {response.text}")

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

def predict(model, record):
    # Crear DMatrix con los datos de entrada
    st.write("Datos enviados al modelo para predicción:")
    st.write(record)
    dmatrix = xgb.DMatrix(record)
    prediccion = model.predict(dmatrix)[0]
    return float(prediccion)

# En la sección de predicción:
if calcular_prediccion or submit_new:
    # Convertir el registro a DataFrame y codificar
    input_df = pd.DataFrame([input_data])
    encoded_df = pd.get_dummies(input_df, columns=["clima", "nivel_trafico", "momento_del_dia", "tipo_vehiculo"])
    
    # Asegurar que todas las columnas esperadas estén presentes
    for col in model.feature_names:
        if col not in encoded_df.columns:
            encoded_df[col] = 0

    # Reorganizar las columnas para que coincidan con el modelo
    encoded_df = encoded_df[model.feature_names]
    
    # **Imprimir datos para depuración**
    st.write("Datos después de la codificación y alineación:")
    st.write(encoded_df)
    
    # Realizar la predicción
    tiempo_predicho = predict(model, encoded_df)
    st.write(f"**Tiempo estimado de entrega:** {tiempo_predicho:.2f} minutos")


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

            calcular_prediccion = st.form_submit_button("Calcular Predicción")
            submit_new = st.form_submit_button("Crear Pedido")

            input_data = {
                "distancia_km": distancia_km,
                "tiempo_preparacion_min": tiempo_preparacion_min,
                "experiencia_repartidor_anos": experiencia_repartidor_anos,
                "clima": clima,
                "nivel_trafico": nivel_trafico,
                "momento_del_dia": momento_del_dia,
                "tipo_vehiculo": tipo_vehiculo,
            }

            if calcular_prediccion:
                input_df = pd.DataFrame([input_data])
                encoded_df = pd.get_dummies(input_df, columns=["clima", "nivel_trafico", "momento_del_dia", "tipo_vehiculo"])
                for col in model.feature_names:
                    if col not in encoded_df.columns:
                        encoded_df[col] = 0
                encoded_df = encoded_df[model.feature_names]
                tiempo_predicho = predict(model, encoded_df)
                st.write(f"**Tiempo estimado de entrega:** {tiempo_predicho:.2f} minutos")

            if submit_new:
                input_df = pd.DataFrame([input_data])
                encoded_df = pd.get_dummies(input_df, columns=["clima", "nivel_trafico", "momento_del_dia", "tipo_vehiculo"])
                for col in model.feature_names:
                    if col not in encoded_df.columns:
                        encoded_df[col] = 0
                encoded_df = encoded_df[model.feature_names]
                tiempo_predicho = predict(model, encoded_df)
                input_data["tiempo_entrega_min"] = tiempo_predicho
                insert_data_to_supabase(input_data)

    # Tab: Modificar Pedido
    with tab2:
        st.header("Modificar Pedido")
        pedidos = fetch_pedidos()
        if not pedidos.empty:
            st.dataframe(pedidos)
            pedido_id = st.selectbox("Selecciona un ID para modificar", pedidos["id"].tolist())
            selected_pedido = pedidos[pedidos["id"] == pedido_id].iloc[0].to_dict()
            with st.form("Modificar Pedido"):
                distancia_km = st.number_input("Distancia (km)", value=float(selected_pedido["distancia_km"]), min_value=0.0, step=0.1)
                tiempo_preparacion_min = st.number_input("Tiempo de preparación (min)", value=int(selected_pedido["tiempo_preparacion_min"]), min_value=0, step=1)
                experiencia_repartidor_anos = st.number_input("Experiencia del repartidor (años)", value=float(selected_pedido["experiencia_repartidor_anos"]), min_value=0.0, step=0.1)
                clima = st.selectbox("Clima", ["Despejado", "Lluvioso", "Ventoso", "Niebla"], index=["Despejado", "Lluvioso", "Ventoso", "Niebla"].index(selected_pedido["clima"]))
                nivel_trafico = st.selectbox("Nivel de tráfico", ["Bajo", "Medio", "Alto"], index=["Bajo", "Medio", "Alto"].index(selected_pedido["nivel_trafico"]))
                momento_del_dia = st.selectbox("Momento del día", ["Mañana", "Tarde", "Noche", "Madrugada"], index=["Mañana", "Tarde", "Noche", "Madrugada"].index(selected_pedido["momento_del_dia"]))
                tipo_vehiculo = st.selectbox("Tipo de vehículo", ["Bicicleta", "Patineta", "Moto", "Auto"], index=["Bicicleta", "Patineta", "Moto", "Auto"].index(selected_pedido["tipo_vehiculo"]))

                calcular_prediccion_mod = st.form_submit_button("Calcular Predicción")
                submit_update = st.form_submit_button("Actualizar Pedido")

                updated_data = {
                    "distancia_km": distancia_km,
                    "tiempo_preparacion_min": tiempo_preparacion_min,
                    "experiencia_repartidor_anos": experiencia_repartidor_anos,
                    "clima": clima,
                    "nivel_trafico": nivel_trafico,
                    "momento_del_dia": momento_del_dia,
                    "tipo_vehiculo": tipo_vehiculo,
                }

                if calcular_prediccion_mod:
                    updated_df = pd.DataFrame([updated_data])
                    encoded_df = pd.get_dummies(updated_df, columns=["clima", "nivel_trafico", "momento_del_dia", "tipo_vehiculo"])
                    for col in model.feature_names:
                        if col not in encoded_df.columns:
                            encoded_df[col] = 0
                    encoded_df = encoded_df[model.feature_names]
                    tiempo_predicho = predict(model, encoded_df)
                    st.write(f"**Tiempo estimado de entrega:** {tiempo_predicho:.2f} minutos")
                    updated_data["tiempo_entrega_min"] = tiempo_predicho

                if submit_update:
                    update_pedido(pedido_id, updated_data)

    # Tab: Borrar Pedido
    with tab3:
        st.header("Borrar Pedido")
        pedidos = fetch_pedidos()
        if not pedidos.empty:
            st.dataframe(pedidos)
            pedido_id = st.selectbox("Selecciona un ID para borrar", pedidos["id"].tolist())
            if st.button("Eliminar Pedido"):
                delete_pedido(pedido_id)

    # Tab: Buscar Pedido
    with tab4:
        st.header("Buscar Pedido")
        pedidos = fetch_pedidos()
        if not pedidos.empty:
            search_id = st.text_input("Buscar por ID")
            if search_id:
                search_result = pedidos[pedidos["id"] == int(search_id)]
                if not search_result.empty:
                    st.dataframe(search_result)
                else:
                    st.warning("No se encontró ningún pedido con ese ID.")
