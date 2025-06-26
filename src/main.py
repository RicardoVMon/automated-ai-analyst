import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

from modules.prompts import relacion_semantica
from modules.load import leer_archivos
from modules.preprocessing import clasificar_variables

st.set_page_config(page_title="Análisis de Datos", layout="wide", initial_sidebar_state="expanded")

def cargar_api_gemini():
    try:
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        st.error(f"Error al cargar la API de Gemini: {e}")
        return None
    
# --- Configuración inicial ---
st.title("Análisis de Datos Automátizado")

# --- API Key y modelo ---
model = cargar_api_gemini()

# --- Carga de archivos ---
uploaded_files = st.file_uploader(
    "Carga tu archivo CSV o Excel", 
    type=["csv", "xlsx"], 
    accept_multiple_files=True
)

# Filtrar archivos duplicados por nombre y tamaño
if uploaded_files:
    unique_files = []
    seen = set()
    for file in uploaded_files:
        file_id = (file.name, file.size)
        if file_id not in seen:
            unique_files.append(file)
            seen.add(file_id)
        else:
            st.warning(f"Archivo duplicado ignorado: {file.name}")

    uploaded_files = unique_files

# Carga inicial y análisis semántico de múltiples archivos
if uploaded_files:
    dataframes, nombres = leer_archivos(uploaded_files)

    if len(dataframes) > 0:
        st.success(f"{len(dataframes)} archivo(s) cargado(s) correctamente")

        # --- Vista previa de archivos ---
        st.subheader("Vista previa de archivos")
        for nombre, df in zip(nombres, dataframes):
            st.markdown(f"### {nombre}")
            st.dataframe(df.head(), use_container_width=True)

        _ = """        ############# Clasificación de variables por archivo, hay que ponerla en otro lado más significativo #############
         st.subheader("Clasificación de variables por archivo")
        for nombre, df in zip(nombres, dataframes):
            st.markdown(f"#### {nombre}")
            
            # Crear clave única para cada archivo
            clasificacion_clave = f"clasificacion_{nombre}_{len(df.columns)}_{hash(str(df.columns.tolist()))}"
            
            # Solo ejecutar clasificación si no se ha hecho antes
            if clasificacion_clave not in st.session_state:
                with st.spinner("Clasificando variables..."):
                    try:
                        st.session_state[clasificacion_clave] = clasificar_variables(df, model)
                    except Exception as e:
                        st.error(f"Error al clasificar variables de {nombre}: {e}")
                        st.session_state[clasificacion_clave] = {}
            
            # Usar el resultado almacenado
            clasif = st.session_state.get(clasificacion_clave, {})
            
            if clasif:
                st.json(clasif)
                
                # Botón para volver a clasificar si se desea
                if st.button(f"Volver a clasificar {nombre}", key=f"reclasificar_{nombre}"):
                    if clasificacion_clave in st.session_state:
                        del st.session_state[clasificacion_clave]
                    st.rerun()
            else:
                st.warning(f"No se pudo clasificar las variables de {nombre}") """

        # --- Si hay más de un archivo, permitir análisis semántico ---
        if len(dataframes) > 1:
            st.subheader("Análisis semántico de relación entre archivos")
            
            # Crear una clave única para el estado basada en los nombres de archivos
            archivos_clave = "_".join(sorted(nombres))
            estado_clave = f"analisis_semantico_{archivos_clave}"
            
            # Solo ejecutar el análisis si no se ha hecho antes
            if estado_clave not in st.session_state:
                with st.spinner("Analizando relación con Gemini..."):
                    try:
                        st.session_state[estado_clave] = relacion_semantica(dataframes, model)
                    except Exception as e:
                        st.error(f"Error al usar la API de Gemini: {e}")
                        st.session_state[estado_clave] = None
            
            # Usar el resultado almacenado
            relacionados = st.session_state.get(estado_clave)
            
            if relacionados is True:
                st.success("Los archivos están relacionados semánticamente, se usarán para el análisis de datos.")
            elif relacionados is False:
                st.error(f"Al menos un archivo no está relacionado semánticamente, se analizará el último archivo ingresado: **{nombres[-1]}**.")
                
                # Botón para limpiar caché si se quiere volver a analizar
                if st.button("Volver a analizar relación semántica", key="reanalizar"):
                    if estado_clave in st.session_state:
                        del st.session_state[estado_clave]
                    st.rerun()
            else:
                st.warning("No se pudo determinar claramente la relación.")
    else:
        st.warning("No se pudo cargar ningún archivo válido.")
else:
    st.info("Esperando archivos...")

