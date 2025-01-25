import streamlit as st
import pandas as pd
import requests

# Configuración de Supabase
SUPABASE_URL = "https://rtporjxjyrkttnvjtqmg.supabase.co"  # Reemplaza con tu URL de Supabase
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ0cG9yanhqeXJrdHRudmp0cW1nIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjY2OTEzNDAsImV4cCI6MjA0MjI2NzM0MH0.ghyQtdPB-db6_viDlJlQDLDL_h7tAukRWycVyfAE6zk"  # Reemplaza con tu API Key
TABLE_NAME = "datos_crudos"  # Nombre de tu tabla en Supabase

# Valores predeterminados para reemplazar nulos
DEFAULT_VALUES = {
    "ID_Pedido": 0,                   # Valor predeterminado para ID_Pedido
    "Distancia_km": 0.0,              # Valor predeterminado para Distancia_km
    "Clima": "Desconocido",           # Valor predeterminado para Clima
    "Nivel_Trafico": "Desconocido",   # Valor predeterminado para Nivel_Trafico
    "Momento_Del_Dia": "Desconocido", # Valor predeterminado para Momento_Del_Dia
    "Tipo_Vehiculo": "Desconocido",   # Valor predeterminado para Tipo_Vehiculo
    "Tiempo_Preparacion_min": 0,      # Valor predeterminado para Tiempo_Preparacion_min
    "Experiencia_Repartidor_anos": 0, # Valor predeterminado para Experiencia_Repartidor_anos
    "Tiempo_Entrega_min": 0           # Valor predeterminado para Tiempo_Entrega_min
}

# Función para cargar datos a Supabase
def upload_to_supabase(dataframe):
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Reemplazar valores nulos en todas las columnas con valores predeterminados
    dataframe = dataframe.fillna(DEFAULT_VALUES)
    
    # Convertir cada fila del DataFrame a JSON
    try:
        data = dataframe.to_dict(orient="records")
    except Exception as e:
        return False, f"Error al preparar los datos: {str(e)}"
    
    # Enviar los datos a Supabase
    try:
        response = requests.post(f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}", headers=headers, json=data)
        if response.status_code == 201:
            return True, "Datos subidos exitosamente."
        else:
            return False, f"Error al subir datos: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Error de conexión: {str(e)}"

# Interfaz de Streamlit
st.title("Subir Documento y Guardar en Supabase")
st.write("Sube un archivo CSV para cargarlo directamente en la base de datos de Supabase.")

# Subir archivo
uploaded_file = st.file_uploader("Elige un archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer el archivo CSV
    dataframe = pd.read_csv(uploaded_file)
    st.write("Previsualización de los datos cargados:")
    st.dataframe(dataframe)
    
    # Botón para subir los datos a Supabase
    if st.button("Subir a Supabase"):
        success, message = upload_to_supabase(dataframe)
        if success:
            st.success(message)
        else:
            st.error(message)
