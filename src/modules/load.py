import streamlit as st
import pandas as pd
from modules.prompts import prompt_relacion_semantica

# --- Función para leer el archivo ---
def leer_archivo(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Tipo de archivo no compatible. Usa CSV o Excel.")
            return None
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return None
# --- Función para leer múltiples archivos ---
def leer_archivos(lista_archivos):
    dataframes = []
    nombres = []
    for file in lista_archivos:
        df = leer_archivo(file)
        if df is not None:
            dataframes.append(df)
            nombres.append(file.name)
    return dataframes, nombres

def relacion_semantica(dataframes, model):
    prompt = prompt_relacion_semantica(dataframes)
    response = model.generate_content(prompt)
    texto = response.text.strip().lower()
    if "sí" in texto or "si" in texto:
        return True
    elif "no" in texto:
        return False
    else:
        # Si la respuesta no es clara, mejor asumir False o manejar el caso
        return None