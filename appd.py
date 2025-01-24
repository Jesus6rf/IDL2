import streamlit as st
import pandas as pd
import requests
import json
import pickle
from io import BytesIO
from supabase import create_client, Client

# Configuración de Supabase
SUPABASE_URL = "https://<tu-supabase-url>"  # Cambia por tu URL de Supabase
SUPABASE_KEY = "<tu-supabase-key>"  # Cambia por tu clave de API
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Función para cargar el modelo desde Supabase
@st.cache_resource
def cargar_modelo():
    response = supabase.storage.from_("modelos").download("random_forest_model.pkl")
    model_data = BytesIO(response)
    return pickle.load(model_data)

# Cargar el modelo al iniciar
modelo = cargar_modelo()

# Función para leer registros desde Supabase
def leer_registros():
    data = supabase.table("Registro_Datos").select("*").execute()
    return pd.DataFrame(data.data)

# Función para agregar un nuevo registro
def agregar_registro(registro):
    supabase.table("Registro_Datos").insert(registro).execute()

# Función para actualizar un registro
def actualizar_registro(id_pedido, valores_actualizados):
    supabase.table("Registro_Datos").update(valores_actualizados).eq("ID_Pedido", id_pedido).execute()

# Función para eliminar un registro
def eliminar_registro(id_pedido):
    supabase.table("Registro_Datos").delete().eq("ID_Pedido", id_pedido).execute()

# Opciones de listas desplegables
clima_opciones = ["Despejado", "Lluvioso", "Nublado", "Ventoso"]
nivel_trafico_opciones = ["Bajo", "Medio", "Alto"]
momento_dia_opciones = ["Mañana", "Tarde", "Noche"]
tipo_vehiculo_opciones = ["Bicicleta", "Moto", "Coche", "Patineta"]

# Aplicación Streamlit
st.title("Gestión de Datos y Predicciones")

# Configuración de pestañas
tabs = st.tabs(["Leer Registros", "Agregar Registro", "Actualizar Registro", "Eliminar Registro", "Predicción"])

# Pestaña: Leer Registros
with tabs[0]:
    st.subheader("Registros en Supabase")
    registros = leer_registros()
    if not registros.empty:
        st.dataframe(registros)
    else:
        st.warning("No hay registros disponibles.")

# Pestaña: Agregar Registro
with tabs[1]:
    st.subheader("Agregar un nuevo registro")
    distancia = st.number_input("Distancia (km)", min_value=0.0)
    clima = st.selectbox("Clima", options=clima_opciones)
    nivel_trafico = st.selectbox("Nivel de Tráfico", options=nivel_trafico_opciones)
    momento_dia = st.selectbox("Momento del Día", options=momento_dia_opciones)
    tipo_vehiculo = st.selectbox("Tipo de Vehículo", options=tipo_vehiculo_opciones)
    tiempo_preparacion = st.number_input("Tiempo de Preparación (min)", min_value=0)
    experiencia = st.number_input("Experiencia del Repartidor (años)", min_value=0.0)
    if st.button("Agregar Registro"):
        nuevo_registro = {
            "Distancia_km": distancia,
            "Clima": clima_opciones.index(clima),
            "Nivel_Trafico": nivel_trafico_opciones.index(nivel_trafico),
            "Momento_Del_Dia": momento_dia_opciones.index(momento_dia),
            "Tipo_Vehiculo": tipo_vehiculo_opciones.index(tipo_vehiculo),
            "Tiempo_Preparacion_min": tiempo_preparacion,
            "Experiencia_Repartidor_anos": experiencia,
            "Tiempo_Entrega_min": None,  # Se calculará con el modelo
        }
        agregar_registro(nuevo_registro)
        st.success("Registro agregado correctamente")

# Pestaña: Actualizar Registro
with tabs[2]:
    st.subheader("Actualizar un registro existente")
    id_actualizar = st.number_input("ID del Pedido", min_value=0, step=1)
    distancia = st.number_input("Nueva Distancia (km)", min_value=0.0)
    tiempo_preparacion = st.number_input("Nuevo Tiempo de Preparación (min)", min_value=0)
    if st.button("Actualizar Registro"):
        valores_actualizados = {
            "Distancia_km": distancia,
            "Tiempo_Preparacion_min": tiempo_preparacion,
        }
        actualizar_registro(id_actualizar, valores_actualizados)
        st.success("Registro actualizado correctamente")

# Pestaña: Eliminar Registro
with tabs[3]:
    st.subheader("Eliminar un registro existente")
    id_eliminar = st.number_input("ID del Pedido para eliminar", min_value=0, step=1)
    if st.button("Eliminar Registro"):
        eliminar_registro(id_eliminar)
        st.success("Registro eliminado correctamente")

# Pestaña: Predicción
with tabs[4]:
    st.subheader("Realizar predicción para el último registro")
    registros = leer_registros()
    if not registros.empty:
        ultimo_registro = registros.iloc[-1].drop("Tiempo_Entrega_min").values.reshape(1, -1)
        prediccion = modelo.predict(ultimo_registro)
        st.write(f"Predicción para el último registro: {prediccion[0]} minutos")
    else:
        st.warning("No hay registros disponibles para predicción.")
