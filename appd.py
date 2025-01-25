import streamlit as st
import pandas as pd
import requests
import json
import pickle
from io import BytesIO
from supabase import create_client, Client

# Configuración de Supabase
SUPABASE_URL = "https://rtporjxjyrkttnvjtqmg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ0cG9yanhqeXJrdHRudmp0cW1nIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjY2OTEzNDAsImV4cCI6MjA0MjI2NzM0MH0.ghyQtdPB-db6_viDlJlQDLDL_h7tAukRWycVyfAE6zk"
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
    try:
        data = supabase.table("nuevos_datos").select("*").execute()
        return pd.DataFrame(data.data)
    except Exception as e:
        st.error(f"Error al leer registros: {e}")
        return pd.DataFrame()

# Función para agregar un nuevo registro
def agregar_registro(registro):
    try:
        supabase.table("nuevos_datos").insert(registro).execute()
    except Exception as e:
        st.error(f"Error al agregar el registro: {e}")

# Función para actualizar un registro
def actualizar_registro(id_pedido, valores_actualizados):
    try:
        supabase.table("nuevos_datos").update(valores_actualizados).eq("ID_Pedido", id_pedido).execute()
    except Exception as e:
        st.error(f"Error al actualizar el registro: {e}")

# Función para eliminar un registro
def eliminar_registro(id_pedido):
    try:
        supabase.table("nuevos_datos").delete().eq("ID_Pedido", id_pedido).execute()
    except Exception as e:
        st.error(f"Error al eliminar el registro: {e}")

# Opciones de listas desplegables
clima_opciones = ["Despejado", "Lluvioso", "Nublado", "Ventoso"]
nivel_trafico_opciones = ["Bajo", "Medio", "Alto"]
momento_dia_opciones = ["Mañana", "Tarde", "Noche"]
tipo_vehiculo_opciones = ["Bicicleta", "Moto", "Coche", "Patineta"]

# Aplicación Streamlit
st.title("Gestión de Pedidos y Predicciones")

# Configuración de pestañas
tabs = st.tabs(["Ver Pedidos", "Crear Pedido", "Modificar Pedido", "Eliminar Pedido"])

# Pestaña: Ver Pedidos
with tabs[0]:
    st.subheader("Registros en Supabase")
    registros = leer_registros()
    if not registros.empty:
        st.dataframe(registros)
    else:
        st.warning("No hay registros disponibles.")

# Pestaña: Crear Pedido
with tabs[1]:
    st.subheader("Crear un nuevo pedido")
    distancia = st.number_input("Distancia (km)", min_value=0.0)
    clima = st.selectbox("Clima", options=clima_opciones)
    nivel_trafico = st.selectbox("Nivel de Tráfico", options=nivel_trafico_opciones)
    momento_dia = st.selectbox("Momento del Día", options=momento_dia_opciones)
    tipo_vehiculo = st.selectbox("Tipo de Vehículo", options=tipo_vehiculo_opciones)
    tiempo_preparacion = st.number_input("Tiempo de Preparación (min)", min_value=0)
    experiencia = st.number_input("Experiencia del Repartidor (años)", min_value=0.0)

    if st.button("Crear y Predecir Pedido"):
        nuevo_registro = {
            "Distancia_km": float(distancia),
            "Clima": int(clima_opciones.index(clima)),
            "Nivel_Trafico": int(nivel_trafico_opciones.index(nivel_trafico)),
            "Momento_Del_Dia": int(momento_dia_opciones.index(momento_dia)),
            "Tipo_Vehiculo": int(tipo_vehiculo_opciones.index(tipo_vehiculo)),
            "Tiempo_Preparacion_min": int(tiempo_preparacion),
            "Experiencia_Repartidor_anos": float(experiencia),
        }
        input_modelo = pd.DataFrame([nuevo_registro])
        prediccion = modelo.predict(input_modelo)[0]
        nuevo_registro["Tiempo_Entrega_min"] = int(prediccion)

        agregar_registro(nuevo_registro)
        st.success(f"Pedido creado correctamente. Tiempo estimado de entrega: {prediccion:.2f} minutos.")

# Pestaña: Modificar Pedido
with tabs[2]:
    st.subheader("Modificar un pedido existente")
    registros = leer_registros()
    if not registros.empty:
        id_pedido = st.selectbox("Selecciona el ID del Pedido", registros["ID_Pedido"])
        pedido_seleccionado = registros[registros["ID_Pedido"] == id_pedido].iloc[0]

        distancia = st.number_input("Distancia (km)", min_value=0.0, value=pedido_seleccionado["Distancia_km"])
        clima = st.selectbox("Clima", options=clima_opciones, index=pedido_seleccionado["Clima"])
        nivel_trafico = st.selectbox("Nivel de Tráfico", options=nivel_trafico_opciones, index=pedido_seleccionado["Nivel_Trafico"])
        momento_dia = st.selectbox("Momento del Día", options=momento_dia_opciones, index=pedido_seleccionado["Momento_Del_Dia"])
        tipo_vehiculo = st.selectbox("Tipo de Vehículo", options=tipo_vehiculo_opciones, index=pedido_seleccionado["Tipo_Vehiculo"])
        tiempo_preparacion = st.number_input("Tiempo de Preparación (min)", min_value=0, value=pedido_seleccionado["Tiempo_Preparacion_min"])
        experiencia = st.number_input("Experiencia del Repartidor (años)", min_value=0.0, value=pedido_seleccionado["Experiencia_Repartidor_anos"])

        if st.button("Actualizar y Predecir Pedido"):
            valores_actualizados = {
                "Distancia_km": float(distancia),
                "Clima": int(clima_opciones.index(clima)),
                "Nivel_Trafico": int(nivel_trafico_opciones.index(nivel_trafico)),
                "Momento_Del_Dia": int(momento_dia_opciones.index(momento_dia)),
                "Tipo_Vehiculo": int(tipo_vehiculo_opciones.index(tipo_vehiculo)),
                "Tiempo_Preparacion_min": int(tiempo_preparacion),
                "Experiencia_Repartidor_anos": float(experiencia),
            }
            input_modelo = pd.DataFrame([valores_actualizados])
            prediccion = modelo.predict(input_modelo)[0]
            valores_actualizados["Tiempo_Entrega_min"] = int(prediccion)

            actualizar_registro(id_pedido, valores_actualizados)
            st.success(f"Pedido actualizado correctamente. Tiempo estimado de entrega: {prediccion:.2f} minutos.")
    else:
        st.warning("No hay registros disponibles para modificar.")

# Pestaña: Eliminar Pedido
with tabs[3]:
    st.subheader("Eliminar un pedido existente")
    registros = leer_registros()
    if not registros.empty:
        id_eliminar = st.selectbox("Selecciona el ID del Pedido para eliminar", registros["ID_Pedido"])
        if st.button("Eliminar Pedido"):
            eliminar_registro(id_eliminar)
            st.success("Pedido eliminado correctamente")
    else:
        st.warning("No hay registros disponibles para eliminar.")
