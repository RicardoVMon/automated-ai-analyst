import streamlit as st
import pandas as pd

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
def leer_archivos(lista_archivos, archivos_sesion=None):
    archivos = {}
    archivos_sesion = archivos_sesion or {}
    for file in lista_archivos:
        df = leer_archivo(file)
        if df is not None:
            clasificacion_existente = archivos_sesion.get(file.name, {}).get('clasificacion')
            descripcion_existente = archivos_sesion.get(file.name, {}).get('descripcion')
            archivos[file.name] = {
                'dataframe': df,
                'clasificacion': clasificacion_existente,  # Preservar clasificación existente o None
                'descripcion': descripcion_existente  # Preservar descripción existente o None
            }
    return archivos


def filtrar_duplicados(archivos_cargados):
    unique_files = []
    seen = set()
    for file in archivos_cargados:
        file_id = (file.name, file.size)
        if file_id not in seen:
            unique_files.append(file)
            seen.add(file_id)
        else:
            st.warning(f"Archivo duplicado ignorado: {file.name}")

    return unique_files