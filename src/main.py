import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai

st.set_page_config(page_title="Análisis de Datos", layout="centered")

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

def prompt_clasificacion_variables(df):
    columnas = list(df.columns)
    muestra = df.head(2).to_dict(orient='records')

    prompt = f"""
        Tengo un conjunto de datos con las siguientes columnas:

        {columnas}

        Y estas son dos filas de ejemplo de datos:

        {muestra}

        Por favor, clasifica cada columna en una de las siguientes categorías:

        - Numérica
        - Categórica
        - Temporal
        - Booleana
        - Desconocida

        Devuelve la clasificación en formato:

        Columna1: Tipo
        Columna2: Tipo
        ...

        No agregues nada más.
        """
    return prompt

def clasificar_variables(df, model):
    prompt = prompt_clasificacion_variables(df)
    response = model.generate_content(prompt)
    texto = response.text.strip()

    clasificacion = {}
    for linea in texto.splitlines():
        if ':' in linea:
            col, tipo = linea.split(':', 1)
            clasificacion[col.strip()] = tipo.strip()
    return clasificacion

def prompt_relacion_semantica(dataframes):
    info_archivos = []
    for i, df in enumerate(dataframes, start=1):
        columnas = list(df.columns)
        muestra = df.head(2).to_dict(orient='records')
        info_archivos.append(f"Archivo {i}:\n- Columnas: {columnas}\n- Ejemplos (2 filas): {muestra}")

    info_texto = "\n\n".join(info_archivos)

    prompt = f"""
        Tengo {len(dataframes)} conjuntos de datos representados por {len(dataframes)} archivos. Necesito saber si **alguno de estos archivos está relacionado semánticamente con alguno de los otros**.

        Considera:
        - Relación semántica significa que los datos tienen sentido juntos, pertenecen a un mismo dominio o contexto.
        - No te bases solo en nombres similares o valores iguales, sino en el significado real.
        - Por ejemplo, archivos de perros y archivos de aviones NO están relacionados.
        - Devuélveme solo "Sí" si todos los archivos forman parte de un mismo contexto semántico, o "No" si no tienen relación.

        Aquí está la información resumida de cada archivo:

        {info_texto}

        Responde solo con "Sí" o "No".
        """
    return prompt

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

def cargar_api_gemini():
    try:
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        st.error(f"Error al cargar la API de Gemini: {e}")
        return None

# --- Configuración inicial ---
st.title("Análisis de Datos Inicial")

# --- API Key y modelo ---
model = cargar_api_gemini()

# --- Carga de archivos ---
uploaded_files = st.file_uploader("Carga tu archivo CSV o Excel", type=["csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    dataframes, nombres = leer_archivos(uploaded_files)

    if len(dataframes) > 0:
        st.success(f"{len(dataframes)} archivo(s) cargado(s) correctamente")

        # --- Vista previa de archivos ---
        st.subheader("Vista previa de archivos")
        for nombre, df in zip(nombres, dataframes):
            st.markdown(f"### 📄 {nombre}")
            st.dataframe(df.head(), use_container_width=True)

        if model:
            st.subheader("Clasificación de variables por archivo")
            for nombre, df in zip(nombres, dataframes):
                st.markdown(f"#### {nombre}")
                with st.spinner("Clasificando variables..."):
                    clasif = clasificar_variables(df, model)
                st.json(clasif)

            # --- Si hay más de un archivo, permitir análisis semántico ---
            if len(dataframes) > 1:
                    st.subheader("Análisis semántico de relación entre archivos")
                    with st.spinner("Analizando relación con Gemini..."):
                        try:
                            relacionados = relacion_semantica(dataframes, model)
                            if relacionados is True:
                                st.success("Sí, los archivos están relacionados semánticamente.")
                            elif relacionados is False:
                                st.error("No todos los archivos están relacionados semánticamente.")
                            else:
                                st.warning("No se pudo determinar claramente la relación.")
                        except Exception as e:
                            st.error(f"Error al usar la API de Gemini: {e}")
        elif not model:
            st.warning("Configura la API Key para usar Gemini.")
    else:
        st.warning("No se pudo cargar ningún archivo válido.")
else:
    st.info("Esperando archivos...")