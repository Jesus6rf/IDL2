import streamlit as st
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Configuración de Supabase (usa tu URL y API Key)
SUPABASE_URL = "https://rtporjxjyrkttnvjtqmg.supabase.co"  # Reemplaza con tu URL
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ0cG9yanhqeXJrdHRudmp0cW1nIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjY2OTEzNDAsImV4cCI6MjA0MjI2NzM0MH0.ghyQtdPB-db6_viDlJlQDLDL_h7tAukRWycVyfAE6zk"  # Reemplaza con tu API Key

# Función para cargar datos desde Supabase
@st.cache_data
def load_data():
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
    }
    response = requests.get(f"{SUPABASE_URL}/rest/v1/datos_crudos", headers=headers, params={"select": "*"})
    
    if response.status_code == 200:
        data = pd.DataFrame(response.json())
        return data
    else:
        st.error(f"Error al cargar datos: {response.status_code} - {response.text}")
        return pd.DataFrame()

# Preprocesamiento de datos
def preprocess_data(data):
    # Variables categóricas a codificar
    categorical_cols = ['Clima', 'Nivel_Trafico', 'Momento_Del_Dia', 'Tipo_Vehiculo']
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_cols]), columns=encoder.get_feature_names_out())
    
    # Unir las variables codificadas al dataset original
    data = pd.concat([data.drop(columns=categorical_cols), encoded_data], axis=1)
    return data

# Entrenamiento del modelo
def train_model(data):
    # Preprocesar datos
    data = preprocess_data(data)
    
    # Selección de características y variable objetivo
    X = data.drop(columns=['ID_Pedido', 'Tiempo_Entrega_min'])
    y = data['Tiempo_Entrega_min']
    
    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar un modelo Random Forest
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluación del modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, mse, r2

# Visualización de resultados
def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title('Valores reales vs Predicciones')
    st.pyplot(plt)

# Interfaz de Streamlit
st.title("Predicción del Tiempo de Entrega")
st.write("Cargando datos desde Supabase...")

# Cargar y mostrar datos
data = load_data()
if not data.empty:
    if st.checkbox("Mostrar datos cargados"):
        st.dataframe(data)

    # Entrenar el modelo
    if st.button("Entrenar modelo"):
        st.write("Entrenando el modelo...")
        model, X_test, y_test, y_pred, mse, r2 = train_model(data)
        
        st.write("**Métricas del modelo:**")
        st.write(f"- Error cuadrático medio (MSE): {mse:.2f}")
        st.write(f"- Coeficiente de determinación (R2): {r2:.2f}")
        
        st.write("**Visualización de resultados:**")
        plot_results(y_test, y_pred)
else:
    st.error("No se pudieron cargar los datos.")
