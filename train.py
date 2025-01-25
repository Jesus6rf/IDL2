import streamlit as st
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de Supabase
SUPABASE_URL = "https://rtporjxjyrkttnvjtqmg.supabase.co"  # URL de Supabase
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ0cG9yanhqeXJrdHRudmp0cW1nIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjY2OTEzNDAsImV4cCI6MjA0MjI2NzM0MH0.ghyQtdPB-db6_viDlJlQDLDL_h7tAukRWycVyfAE6zk"  # API Key
TABLE_NAME = "datos_crudos"  # Nombre de la tabla en Supabase

# Función para cargar datos desde Supabase
@st.cache_data
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

# Análisis exploratorio de datos (EDA)
def perform_eda(data):
    st.subheader("Análisis Exploratorio de Datos (EDA)")
    st.write("### Resumen estadístico:")
    st.dataframe(data.describe())

    st.write("### Distribución de la variable objetivo (`Tiempo_Entrega_min`):")
    fig, ax = plt.subplots()
    sns.histplot(data["Tiempo_Entrega_min"], kde=True, ax=ax, color="blue")
    st.pyplot(fig)

    st.write("### Mapa de calor de correlación:")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Entrenamiento del modelo
def train_model(data):
    X = data.drop(columns=["ID_Pedido", "Tiempo_Entrega_min"])
    X = pd.get_dummies(X, drop_first=True)  # Codificar variables categóricas
    y = data["Tiempo_Entrega_min"]

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluar modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2, X_test, y_test, y_pred

# Guardar el modelo como archivo .pkl
def save_model(model):
    with open("modelo_entrenado.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("modelo_entrenado.pkl", "rb") as f:
        st.download_button("Descargar Modelo", f, file_name="modelo_entrenado.pkl")

# Interfaz de Streamlit
st.title("Entrenamiento del Modelo con EDA y Datos desde Supabase")

# Estado de carga de datos
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.data = pd.DataFrame()

# Cargar datos desde Supabase solo si no están cargados
if not st.session_state.data_loaded:
    if st.button("Cargar datos desde Supabase"):
        data = load_data_from_supabase()
        if not data.empty:
            st.session_state.data_loaded = True
            st.session_state.data = data
            st.success("Datos cargados correctamente.")
            st.write("Datos cargados:")
            st.dataframe(data)

# Mostrar análisis exploratorio y entrenar modelo si los datos ya están cargados
if st.session_state.data_loaded:
    st.write("### Datos cargados:")
    st.dataframe(st.session_state.data)

    # Análisis Exploratorio de Datos
    perform_eda(st.session_state.data)

    # Entrenar modelo
    st.subheader("Entrenamiento del Modelo")
    if st.button("Entrenar Modelo"):
        model, mse, r2, X_test, y_test, y_pred = train_model(st.session_state.data)
        st.write("Modelo entrenado exitosamente.")
        st.write(f"**Error cuadrático medio (MSE):** {mse:.2f}")
        st.write(f"**Coeficiente de determinación (R²):** {r2:.2f}")

        # Mostrar una comparación de valores reales vs predichos
        st.write("### Comparación entre valores reales y predichos:")
        comparison = pd.DataFrame({"Real": y_test, "Predicho": y_pred}).reset_index(drop=True)
        st.dataframe(comparison.head())

        # Descargar modelo
        save_model(model)
