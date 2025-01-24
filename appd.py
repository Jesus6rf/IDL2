import streamlit as st
import pandas as pd
import requests
import json
import pickle
from io import BytesIO
from supabase import create_client, Client

# Configuración de Supabase
SUPABASE_URL = "https://aispdrqeugwxfhghzkcd.supabase.co"  # Cambia por tu URL de Supabase
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFpc3BkcnFldWd3eGZoZ2h6a2NkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM2NzkxMzgsImV4cCI6MjA0OTI1NTEzOH0.irvfK6Wdo_OMqU29Bhz941t6-y-Zg-YuIpqXNbM3COU"  # Cambia por tu clave de API
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

# Aplicación Streamlit
st.title("Gestión de Datos y Predicciones")

# Opción para ver todos los registros
if st.sidebar.button("Leer Registros"):
    registros = leer_registros()
    st.subheader("Registros en Supabase")
    st.dataframe(registros)

# Opción para agregar un nuevo registro
st.sidebar.subheader("Agregar Registro")
if st.sidebar.button("Nuevo Registro"):
    distancia = st.sidebar.number_input("Distancia (km)", min_value=0.0)
    clima = st.sidebar.number_input("Clima (codificado)", min_value=0)
    nivel_trafico = st.sidebar.number_input("Nivel de Tráfico (codificado)", min_value=0)
    momento_dia = st.sidebar.number_input("Momento del Día (codificado)", min_value=0)
    tipo_vehiculo = st.sidebar.number_input("Tipo de Vehículo (codificado)", min_value=0)
    tiempo_preparacion = st.sidebar.number_input("Tiempo de Preparación (min)", min_value=0)
    experiencia = st.sidebar.number_input("Experiencia del Repartidor (años)", min_value=0.0)
    if st.sidebar.button("Agregar"):
        nuevo_registro = {
            "Distancia_km": distancia,
            "Clima": clima,
            "Nivel_Trafico": nivel_trafico,
            "Momento_Del_Dia": momento_dia,
            "Tipo_Vehiculo": tipo_vehiculo,
            "Tiempo_Preparacion_min": tiempo_preparacion,
            "Experiencia_Repartidor_anos": experiencia,
            "Tiempo_Entrega_min": None,  # Se calculará con el modelo
        }
        agregar_registro(nuevo_registro)
        st.success("Registro agregado correctamente")

# Opción para actualizar un registro existente
st.sidebar.subheader("Actualizar Registro")
id_actualizar = st.sidebar.number_input("ID del Pedido", min_value=0)
if st.sidebar.button("Actualizar"):
    valores_actualizados = {
        "Distancia_km": st.sidebar.number_input("Nueva Distancia (km)", min_value=0.0),
        "Tiempo_Preparacion_min": st.sidebar.number_input("Nuevo Tiempo de Preparación (min)", min_value=0),
    }
    actualizar_registro(id_actualizar, valores_actualizados)
    st.success("Registro actualizado correctamente")

# Opción para eliminar un registro
st.sidebar.subheader("Eliminar Registro")
id_eliminar = st.sidebar.number_input("ID del Pedido para eliminar", min_value=0)
if st.sidebar.button("Eliminar"):
    eliminar_registro(id_eliminar)
    st.success("Registro eliminado correctamente")

# Opción para realizar una predicción
st.sidebar.subheader("Predicción")
if st.sidebar.button("Realizar Predicción"):
    registros = leer_registros()
    if not registros.empty:
        ultimo_registro = registros.iloc[-1].drop("Tiempo_Entrega_min").values.reshape(1, -1)
        prediccion = modelo.predict(ultimo_registro)
        st.write(f"Predicción para el último registro: {prediccion[0]} minutos")
    else:
        st.warning("No hay registros disponibles para predicción.")
