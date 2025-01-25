import streamlit as st
import pandas as pd
import requests

# Configuración de Supabase
SUPABASE_URL = "https://rtporjxjyrkttnvjtqmg.supabase.co"  # Reemplaza con tu URL de Supabase
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ0cG9yanhqeXJrdHRudmp0cW1nIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjY2OTEzNDAsImV4cCI6MjA0MjI2NzM0MH0.ghyQtdPB-db6_viDlJlQDLDL_h7tAukRWycVyfAE6zk"  # Reemplaza con tu API Key
TABLE_NAME = "datos_crudos"  # Nombre de tu tabla en Supabase

# Función para cargar datos a Supabase
def upload_to_supabase(dataframe):
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Convertir cada fila del DataFrame a JSON asegurando serialización correcta
    try:
        data = dataframe.where(pd.notnull(dataframe), None).to_dict(orient="records")  # Reemplaza NaN por None
    except Exception as e:
        return False, f"Error al convertir datos a JSON: {str(e)}"
    
    response = requests.post(f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}", headers=headers, json=data)
    
    if response.status_code == 201:
        return True, "Datos subidos exitosamente."
    else:
        return False, f"Error al subir datos: {response.status_code} - {response.text}"

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
