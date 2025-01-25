import streamlit as st
import pandas as pd
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
        # Convertir los índices de predicción a texto para guardarlos en la base de datos
        registro["Clima"] = clima_opciones[registro["Clima"]]
        registro["Nivel_Trafico"] = nivel_trafico_opciones[registro["Nivel_Trafico"]]
        registro["Momento_Del_Dia"] = momento_dia_opciones[registro["Momento_Del_Dia"]]
        registro["Tipo_Vehiculo"] = tipo_vehiculo_opciones[registro["Tipo_Vehiculo"]]

        # Insertar el registro en Supabase
        supabase.table("nuevos_datos").insert(registro).execute()
    except Exception as e:
        st.error(f"Error al agregar el registro: {e}")

# Función para actualizar un registro
def actualizar_registro(id_pedido, valores_actualizados):
    try:
        # Convertir los índices de predicción a texto antes de actualizar
        valores_actualizados["Clima"] = clima_opciones[valores_actualizados["Clima"]]
        valores_actualizados["Nivel_Trafico"] = nivel_trafico_opciones[valores_actualizados["Nivel_Trafico"]]
        valores_actualizados["Momento_Del_Dia"] = momento_dia_opciones[valores_actualizados["Momento_Del_Dia"]]
        valores_actualizados["Tipo_Vehiculo"] = tipo_vehiculo_opciones[valores_actualizados["Tipo_Vehiculo"]]

        # Actualizar el registro en Supabase
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
    distancia = st.number_input("Distancia (km)", min_value=0.0, key="crear_distancia")
    clima = st.selectbox("Clima del Pedido", options=clima_opciones, key="crear_clima")
    nivel_trafico = st.selectbox("Nivel de Tráfico del Pedido", options=nivel_trafico_opciones, key="crear_trafico")
    momento_dia = st.selectbox("Momento del Día del Pedido", options=momento_dia_opciones, key="crear_momento")
    tipo_vehiculo = st.selectbox("Tipo de Vehículo del Pedido", options=tipo_vehiculo_opciones, key="crear_vehiculo")
    tiempo_preparacion = st.number_input("Tiempo de Preparación (min)", min_value=0, key="crear_tiempo")
    experiencia = st.number_input("Experiencia del Repartidor (años)", min_value=0.0, key="crear_experiencia")

    if st.button("Crear y Predecir Pedido", key="crear_pedido"):
        nuevo_registro = {
            "Distancia_km": float(distancia),
            "Clima": int(clima_opciones.index(clima)),  # Índice para predicción
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
        id_pedido = st.selectbox("Selecciona el ID del Pedido", registros["ID_Pedido"], key="modificar_id")
        pedido_seleccionado = registros[registros["ID_Pedido"] == id_pedido].iloc[0]

        distancia = st.number_input("Distancia (km)", min_value=0.0, value=float(pedido_seleccionado["Distancia_km"]), key="modificar_distancia")
        clima = st.selectbox("Clima del Pedido", options=clima_opciones, index=clima_opciones.index(pedido_seleccionado["Clima"]), key="modificar_clima")
        nivel_trafico = st.selectbox("Nivel de Tráfico del Pedido", options=nivel_trafico_opciones, index=nivel_trafico_opciones.index(pedido_seleccionado["Nivel_Trafico"]), key="modificar_trafico")
        momento_dia = st.selectbox("Momento del Día del Pedido", options=momento_dia_opciones, index=momento_dia_opciones.index(pedido_seleccionado["Momento_Del_Dia"]), key="modificar_momento")
        tipo_vehiculo = st.selectbox("Tipo de Vehículo del Pedido", options=tipo_vehiculo_opciones, index=tipo_vehiculo_opciones.index(pedido_seleccionado["Tipo_Vehiculo"]), key="modificar_vehiculo")
        tiempo_preparacion = st.number_input("Tiempo de Preparación (min)", min_value=0, value=int(pedido_seleccionado["Tiempo_Preparacion_min"]), key="modificar_tiempo")
        experiencia = st.number_input("Experiencia del Repartidor (años)", min_value=0.0, value=float(pedido_seleccionado["Experiencia_Repartidor_anos"]), key="modificar_experiencia")

        if st.button("Actualizar y Predecir Pedido", key="modificar_pedido"):
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
        id_eliminar = st.selectbox("Selecciona el ID del Pedido para eliminar", registros["ID_Pedido"], key="eliminar_id")
        if st.button("Eliminar Pedido", key="eliminar_pedido"):
            eliminar_registro(id_eliminar)
            st.success("Pedido eliminado correctamente")
    else:
        st.warning("No hay registros disponibles para eliminar.")
