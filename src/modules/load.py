import streamlit as st  # Importa Streamlit para mostrar mensajes en la interfaz
import pandas as pd     # Importa pandas para manipulación de datos

# --- Función para leer un archivo individual ---
def leer_archivo(uploaded_file):
    """
    Lee un archivo cargado (CSV o Excel) y lo convierte en un DataFrame de pandas.

    Parámetros:
        uploaded_file: Un archivo cargado por el usuario (Streamlit UploadedFile).

    Retorna:
        Un DataFrame de pandas si la lectura es exitosa, o None si ocurre un error o el tipo de archivo no es compatible.
    """
    try:
        if uploaded_file.name.endswith(".csv"):  # Si el archivo es CSV
            return pd.read_csv(uploaded_file)    # Lee el archivo como CSV y retorna el DataFrame
        elif uploaded_file.name.endswith(".xlsx"):  # Si el archivo es Excel
            return pd.read_excel(uploaded_file)     # Lee el archivo como Excel y retorna el DataFrame
        else:
            st.error("Tipo de archivo no compatible. Usa CSV o Excel.")  # Muestra error si el tipo no es compatible
            return None  # Retorna None si el tipo no es compatible
    except Exception as e:  # Si ocurre algún error al leer el archivo
        st.error(f"Error al leer el archivo: {e}")  # Muestra el error en la interfaz
        return None  # Retorna None si hay error

# --- Función para leer múltiples archivos ---
def leer_archivos(lista_archivos, archivos_sesion=None):
    """
    Lee una lista de archivos cargados y retorna un diccionario con información relevante.

    Parámetros:
        lista_archivos: Lista de archivos cargados por el usuario (Streamlit UploadedFile).
        archivos_sesion: Diccionario opcional con información previa de archivos en sesión (clasificación y descripción).

    Retorna:
        Un diccionario donde la clave es el nombre del archivo y el valor es otro diccionario con:
            - 'dataframe': DataFrame de pandas con los datos del archivo.
            - 'clasificacion': Clasificación previa (si existe) o None.
            - 'descripcion': Descripción previa (si existe) o None.
    """
    archivos = {}  # Diccionario para almacenar los archivos leídos
    archivos_sesion = archivos_sesion or {}  # Usa el diccionario de sesión si existe, si no, uno vacío
    for file in lista_archivos:  # Itera sobre cada archivo cargado
        df = leer_archivo(file)  # Lee el archivo y obtiene el DataFrame
        if df is not None:  # Si la lectura fue exitosa
            clasificacion_existente = archivos_sesion.get(file.name, {}).get('clasificacion')  # Recupera clasificación previa si existe
            descripcion_existente = archivos_sesion.get(file.name, {}).get('descripcion')      # Recupera descripción previa si existe
            archivos[file.name] = {  # Almacena la información en el diccionario
                'dataframe': df,
                'clasificacion': clasificacion_existente,  # Preserva clasificación existente o None
                'descripcion': descripcion_existente       # Preserva descripción existente o None
            }
    return archivos  # Retorna el diccionario con los archivos leídos

# --- Función para filtrar archivos duplicados ---
def filtrar_duplicados(archivos_cargados):
    """
    Elimina archivos duplicados de una lista de archivos cargados, considerando nombre y tamaño.

    Parámetros:
        archivos_cargados: Lista de archivos cargados por el usuario (Streamlit UploadedFile).

    Retorna:
        Una lista de archivos únicos (sin duplicados).
        Muestra una advertencia en la interfaz si algún archivo es duplicado.
    """
    unique_files = []  # Lista para almacenar archivos únicos
    seen = set()       # Conjunto para registrar archivos ya vistos (por nombre y tamaño)
    for file in archivos_cargados:  # Itera sobre cada archivo cargado
        file_id = (file.name, file.size)  # Identificador único por nombre y tamaño
        if file_id not in seen:  # Si el archivo no ha sido visto antes
            unique_files.append(file)  # Agrega el archivo a la lista de únicos
            seen.add(file_id)          # Marca el archivo como visto
        else:
            st.warning(f"Archivo duplicado ignorado: {file.name}")  # Advierte si el archivo es duplicado

    return unique_files  # Retorna la lista