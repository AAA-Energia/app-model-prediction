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
    
def validate_and_adjust_data(data):
    try:
        # Convertir todos los datos a tipo float64
        data = data.astype('float64')
        return data
    except Exception as e:
        st.error(f"Error al convertir los datos: {e}")
        return None

def read_example_csv():
    example_file_path = "data/test.csv"
    example_df = pd.read_csv(example_file_path)
    return example_df

# Interfaz de usuario de Streamlit
st.title("Prueba de Modelos en Producción")

def main():
    st.write("1. Seleccione un modelo")
    model_name = st.radio("", ["Prophet_Model","Gradient_Boosting_Model","random_forest_model"], index = 2)
    #model_stage = st.selectbox("Stage del modelo", ["Production", "Staging", "None"])
    st_frame = st.empty()

    if model_name != "Prophet_Model":
        
        st.write("2. Descargue el archivo y luego llenelo con la información debida")
        example_df = read_example_csv()
        example_csv = example_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar archivo CSV de ejemplo",
            data=example_csv,
            file_name='example.csv',
            mime='text/csv'
        )
        
        st.write("3. Subir archivo CSV con datos segun el archivo de ejemplo")
        uploaded_file = st.file_uploader("", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Datos de prueba:")
            st.write(data)

            #model_name = st.text_input("Nombre del modelo", "Prophet_Model")
            #model_stage = st.selectbox("Stage del modelo", ["Production", "Staging", "None"])

            if st.button("Correr Modelo"):
                model = load_model(model_name)
                st.success(f"Modelo {model_name} cargado exitosamente.")
                #predictions = model.predict(data)
                #st.write("Predicciones:")
                #st.write(predictions)

                valid_data = validate_and_adjust_data(data)

                if valid_data is not None:
                    #st.write("Datos de prueba después de la validación y ajuste:")
                    #st.write(valid_data)

                    try:
                        predictions = model.predict(valid_data)
                        st.write("Predicciones:")
                        st.write(predictions)
                    except Exception as e:
                        st.error(f"Error al ejecutar las predicciones: {e}")

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