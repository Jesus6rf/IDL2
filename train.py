import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Configuración de Supabase
SUPABASE_URL = "https://rtporjxjyrkttnvjtqmg.supabase.co"  # Reemplaza con tu URL de Supabase
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ0cG9yanhqeXJrdHRudmp0cW1nIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjY2OTEzNDAsImV4cCI6MjA0MjI2NzM0MH0.ghyQtdPB-db6_viDlJlQDLDL_h7tAukRWycVyfAE6zk"  # Reemplaza con tu API Key
TABLE_NAME = "datos_crudos"

# Valores predeterminados para reemplazar nulos
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

# Función para limpieza e imputación de datos
def clean_data(dataframe):
    dataframe = dataframe.fillna(DEFAULT_VALUES)
    return dataframe

# Análisis exploratorio de datos
def perform_eda(dataframe):
    st.subheader("Análisis Exploratorio de Datos (EDA)")
    st.write("Resumen estadístico:")
    st.write(dataframe.describe())
    
    st.write("Distribución de la variable objetivo (`Tiempo_Entrega_min`):")
    fig, ax = plt.subplots()
    sns.histplot(dataframe["Tiempo_Entrega_min"], kde=True, ax=ax)
    st.pyplot(fig)

    st.write("Correlación entre variables:")
    corr = dataframe.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Entrenamiento del modelo
def train_model(dataframe):
    X = dataframe.drop(columns=["ID_Pedido", "Tiempo_Entrega_min"])
    X = pd.get_dummies(X, drop_first=True)  # Codificación de variables categóricas
    y = dataframe["Tiempo_Entrega_min"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, mse, r2

# Descarga del modelo en formato .pkl
def save_model(model):
    with open("modelo_entrenado.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("Modelo guardado como `modelo_entrenado.pkl`.")
    with open("modelo_entrenado.pkl", "rb") as f:
        st.download_button("Descargar Modelo", f, file_name="modelo_entrenado.pkl")

# Interfaz de Streamlit
st.title("Pipeline Completo: Limpieza, EDA, Entrenamiento y Exportación del Modelo")

# Cargar datos
st.subheader("Cargar Datos")
if st.button("Cargar datos desde Supabase"):
    data = load_data_from_supabase()
    if not data.empty:
        st.success("Datos cargados correctamente.")
        st.dataframe(data)

        # Limpieza e imputación
        st.subheader("Limpieza e Imputación de Datos")
        cleaned_data = clean_data(data)
        st.write("Datos después de la limpieza:")
        st.dataframe(cleaned_data)

        # Análisis exploratorio
        perform_eda(cleaned_data)

        # Entrenamiento del modelo
        st.subheader("Entrenamiento del Modelo")
        if st.button("Entrenar Modelo"):
            model, X_test, y_test, y_pred, mse, r2 = train_model(cleaned_data)
            st.write(f"**Métricas del Modelo:**")
            st.write(f"- MSE: {mse:.2f}")
            st.write(f"- R²: {r2:.2f}")
            
            # Mostrar predicciones
            st.write("Comparación entre valores reales y predichos:")
            comparison = pd.DataFrame({"Real": y_test, "Predicho": y_pred}).reset_index(drop=True)
            st.write(comparison.head())

            # Guardar el modelo
            save_model(model)
