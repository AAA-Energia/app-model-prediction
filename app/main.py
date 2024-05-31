import streamlit as st
import mlflow
import pandas as pd
from mlflow.exceptions import RestException, MlflowException
import numpy as np
import datetime

st.title("Modelo para la Predicción de la Demanda de Energía")
st.write("Espacio para probar el modelo seleccionado, para la materia de Aprendizaje Automático Aplicado.")
st.write("¿Que pasa con este modelo?")

# Configuración inicial usando secretos de Streamlit
mlflow.set_tracking_uri(st.secrets["MLFLOW_TRACKING_URI"])

# Función para cargar el modelo desde MLflow
def load_model(model_name, model_stage=None):
    model_uri = f"models:/{model_name}@champion"
    try:
        if model_name == "Prophet_Model":
            model = mlflow.prophet.load_model(model_uri)
            return model
        
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except MlflowException as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Interfaz de usuario de Streamlit
st.title("Prueba de Modelos en Producción")

def main():
    # Subir archivo CSV con datos de prueba
    model_name = st.radio("Seleccione un modelo", ["Prophet_Model","Gradient_Boosting_Model","random_forest_model"], index = 2)
    #model_stage = st.selectbox("Stage del modelo", ["Production", "Staging", "None"])
    st_frame = st.empty()

    if model_name != "Prophet_Model":
        uploaded_file = st.file_uploader("Subir archivo CSV con datos de prueba", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Datos de prueba:")
            st.write(data)

            #model_name = st.text_input("Nombre del modelo", "Prophet_Model")
            #model_stage = st.selectbox("Stage del modelo", ["Production", "Staging", "None"])

            if st.button("Cargar Modelo"):
                model = load_model(model_name)
                st.success(f"Modelo {model_name} cargado exitosamente.")
                predictions = model.predict(data)
                st.write("Predicciones:")
                st.write(predictions)
                #if model:
                    
                #if st.button("Ejecutar predicciones"):
    else:
        fecha_inicio = st.date_input("Fecha de inicio de la prediccion:")
        hora_inicio = st.time_input("Hora de inicio de la prediccion:", value=None, step=3600)
        print(hora_inicio)

        forecast_hours = st.number_input("¿A cuántas horas a futuro le gustaria pronosticar?", min_value = 1, max_value = 10, value = 5, step = 1)
        model = load_model(model_name)

        if hora_inicio is not None:
            forecast_df = pd.DataFrame({
                "ds": pd.date_range(datetime.datetime.combine(fecha_inicio,hora_inicio), periods=forecast_hours, freq='h')
            })
            forecast = model.predict(forecast_df)
            pred = pd.DataFrame()

            pred["fecha_y_hora"] = forecast_df['ds']
            pred["demanda"] = forecast["yhat"].tail(forecast_hours)
            
            st.write("Predicciones:")
            st.write(pred[["fecha_y_hora","demanda"]].tail(forecast_hours))
                
if __name__ == "__main__":
    main()