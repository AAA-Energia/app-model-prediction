import streamlit as st
import mlflow
import pandas as pd
from mlflow.exceptions import RestException, MlflowException

st.title("Modelo para la Predicción de la Demanda de Energía")
st.write("Espacio para probar el modelo seleccionado, para la materia de Aprendizaje Automático Aplicado.")
st.write("¿Que pasa con este modelo?")

# Configuración inicial usando secretos de Streamlit
mlflow.set_tracking_uri(st.secrets["MLFLOW_TRACKING_URI"])

# Función para cargar el modelo desde MLflow
def load_model(model_name, model_stage):
    model_uri = f"models:/{model_name}/{model_stage}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except MlflowException as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Interfaz de usuario de Streamlit
st.title("Prueba de Modelos en Producción")

def main():
    # Subir archivo CSV con datos de prueba
    uploaded_file = st.file_uploader("Subir archivo CSV con datos de prueba", type=["csv"])
    st_frame = st.empty()

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Datos de prueba:")
        st.write(data)

        model_name = st.text_input("Nombre del modelo", "Prophet_Model")
        model_stage = st.selectbox("Stage del modelo", ["Production", "Staging", "None"])

        if st.button("Cargar Modelo"):
            model = load_model(model_name, model_stage)
            st.success(f"Modelo {model_name} cargado exitosamente desde {model_stage}.")
            predictions = model.predict(data)
            st.write("Predicciones:")
            st.write(predictions)
            #if model:
                
            #if st.button("Ejecutar predicciones"):
                
if __name__ == "__main__":
    main()