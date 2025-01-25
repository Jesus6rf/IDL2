import streamlit as st
import pandas as pd
import requests
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de Supabase
SUPABASE_URL = "https://rtporjxjyrkttnvjtqmg.supabase.co"  # Reemplaza con tu URL de Supabase
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ0cG9yanhqeXJrdHRudmp0cW1nIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjY2OTEzNDAsImV4cCI6MjA0MjI2NzM0MH0.ghyQtdPB-db6_viDlJlQDLDL_h7tAukRWycVyfAE6zk"  # Reemplaza con tu API Key
TABLE_NAME = "datos_crudos"  # Nombre de tu tabla en Supabase

# Valores predeterminados para imputación de datos
DEFAULT_VALUES = {
    "ID_Pedido": 0,
    "Distancia_km": 0.0,
    "Clima": "Desconocido",
    "Nivel_Trafico": "Desconocido",
    "Momento_Del_Dia": "Desconocido",
    "Tipo_Vehiculo": "Desconocido",
    "Tiempo_Preparacion_min": 0,
    "Experiencia_Repartidor_anos": 0,
    "Tiempo_Entrega_min": 0
}

# Función para cargar datos desde Supabase
def load_data_from_supabase():
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
    }
    response = requests.get(f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}", headers=headers, params={"select": "*"})
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error(f"Error al cargar datos: {response.status_code} - {response.text}")
        return pd.DataFrame()

# Limpieza e imputación de datos
def clean_data(dataframe):
    return dataframe.fillna(DEFAULT_VALUES)

# Entrenamiento del modelo
def train_model(dataframe):
    # Seleccionar características y variable objetivo
    X = dataframe.drop(columns=["ID_Pedido", "Tiempo_Entrega_min"])
    X = pd.get_dummies(X, drop_first=True)  # Codificar variables categóricas
    y = dataframe["Tiempo_Entrega_min"]
    
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

# Guardar el modelo como archivo .pkl
def save_model(model):
    with open("modelo_entrenado.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("modelo_entrenado.pkl", "rb") as f:
        st.download_button("Descargar Modelo", f, file_name="modelo_entrenado.pkl")

# Interfaz de Streamlit
st.title("Entrenamiento del Modelo con Datos de Supabase")

# Cargar datos desde Supabase
if st.button("Cargar datos desde Supabase"):
    data = load_data_from_supabase()
    if not data.empty:
        st.success("Datos cargados correctamente.")
        st.write("Datos crudos cargados:")
        st.dataframe(data)

        # Limpieza e imputación de datos
        cleaned_data = clean_data(data)
        st.write("Datos después de la limpieza e imputación:")
        st.dataframe(cleaned_data)

        # Entrenar modelo
        if st.button("Entrenar Modelo"):
            model, mse, r2 = train_model(cleaned_data)
            st.write("Modelo entrenado exitosamente.")
            st.write(f"**Error cuadrático medio (MSE):** {mse:.2f}")
            st.write(f"**Coeficiente de determinación (R²):** {r2:.2f}")

            # Descargar modelo
            save_model(model)
