import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd

from modules.load import filtrar_duplicados, leer_archivos
from modules.preprocessing import clasificar_variables, describir_archivo
import ydata_profiling

st.set_page_config(page_title="Análista de Datos", layout="centered", initial_sidebar_state="expanded")

def cargar_api_gemini():
    try:
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        st.error(f"Error al cargar la API de Gemini: {e}")
        return None



descripcion_safe = """

Este conjunto de datos parece describir las características de diferentes coches. Se incluyen medidas de rendimiento como millas por galón (mpg), caballos de fuerza (hp) y tiempo de aceleración de 0 a 60 millas por hora (time-to-60). También se proporcionan detalles sobre las especificaciones del motor, como el número de cilindros y pulgadas cúbicas. El peso del vehículo (weightlbs) se registra junto con el año del modelo y la marca del coche. Esto permite un análisis de cómo estas características se relacionan entre sí y cómo han evolucionado a lo largo del tiempo y entre diferentes fabricantes.

#### Descripción de las variables:

- mpg: Millas por galón.
- cylinders: Número de cilindros del motor.
- cubicinches: Cilindrada del motor.
- hp: Caballos de fuerza.
- weightlbs: Peso del coche en libras.
- time-to-60: Tiempo que tarda el coche en acelerar de 0 a 60 millas por hora.
- year: Año del modelo del coche.
- brand: Origen o marca del coche.

"""
clasificacion_safe = {
    "mpg": "Numérica",
    "cylinders": "Numérica",
    "cubicinches": "Numérica",
    "hp": "Numérica",
    "weightlbs": "Numérica",
    "time-to-60": "Numérica",
    "year": "Temporal",
    "brand": "Categórica"
}


# --- Configuración inicial ---
st.title("Análisis de Datos Automátizado")

# --- API Key y modelo ---
model = cargar_api_gemini()

# --- SECCIÓN DE CARGA DE ARCHIVOS ---

# --- Carga de archivos ---
uploaded_files = st.file_uploader(
    "Carga tu archivo CSV o Excel", 
    type=["csv", "xlsx"], 
    accept_multiple_files=True
)

# Selección de archivos
if uploaded_files:
    uploaded_files = filtrar_duplicados(uploaded_files)
    archivos = leer_archivos(uploaded_files, st.session_state.get('archivos', {}))  # ahora retorna {'archivo.csv': {'dataframe': df, 'clasificacion': None}}

    if len(archivos) > 0:
        st.success(f"{len(archivos)} archivo(s) cargado(s) correctamente")

        # Si hay más de un archivo, permitir selección
        if len(archivos) > 1:
            st.subheader("Selecciona los archivos para analizar")

            st.markdown(
                """
                A continuación, seleccione los archivos que desea incluir en el análisis conjunto.

                **Recomendación:** Seleccione archivos que estén relacionados entre sí, es decir, que compartan claves comunes, tengan la misma estructura o puedan combinarse sin problemas.

                **Nota:** Los archivos que no sean seleccionados serán excluidos del análisis.
                """
            )

            seleccionados = st.multiselect(
                "Elige los archivos relacionados:",
                options=list(archivos.keys()),
                default=list(archivos.keys()),
                placeholder="Selecciona uno o varios archivos..."
            )

            if seleccionados:
                # Solo guarda los seleccionados
                st.session_state['archivos'] = {nombre: archivos[nombre] for nombre in seleccionados}
                st.session_state['pasar_a_perfilado'] = True
                st.success(f"Archivos seleccionados para el análisis: {', '.join(seleccionados)}")
            else:
                st.session_state['archivos'] = {}
                st.session_state['pasar_a_perfilado'] = False
                st.warning("No se ha seleccionado ningún archivo relacionado. No se puede continuar con el análisis.")
        else:
            # Si solo hay uno, se guarda directamente
            st.session_state['archivos'] = archivos
            st.session_state['pasar_a_perfilado'] = True

    else:
        st.warning("No se pudo cargar ningún archivo válido.")
else:
    st.session_state['pasar_a_perfilado'] = False
    st.info("Esperando archivos...")


# --- SECCIÓN DE PREPROCESAMIENTO ---

# Perfilado de los datos inicial
if st.session_state.get('pasar_a_perfilado') == True:
    st.title("Preprocesamiento de Datos")
    st.subheader("Clasificación de variables por archivo")

    # Clasificación de variables
    for nombre, datos in st.session_state['archivos'].items():
        df = datos['dataframe']
        st.markdown(f"### {nombre} (Vista Previa)")

        if datos.get('descripcion') is None:
            with st.spinner("Describiendo archivo..."):
                try:
                    # descripcion = describir_archivo(df, model)
                    descripcion = descripcion_safe  # Usar descripción segura para pruebas
                    st.session_state['archivos'][nombre]['descripcion'] = descripcion
                except Exception as e:
                    st.error(f"Error al describir el archivo de {nombre}: {e}")
                    st.session_state['archivos'][nombre]['descripcion'] = {}

        descripcion = st.session_state['archivos'][nombre].get('descripcion', {})

        if datos.get('clasificacion') is None:
            with st.spinner("Clasificando variables..."):
                try:
                    # clasificacion = clasificar_variables(df, model)
                    clasificacion = clasificacion_safe  # Usar clasificación segura para pruebas
                    st.session_state['archivos'][nombre]['clasificacion'] = clasificacion
                except Exception as e:
                    st.error(f"Error al clasificar variables de {nombre}: {e}")
                    st.session_state['archivos'][nombre]['clasificacion'] = {}

        clasificacion = st.session_state['archivos'][nombre].get('clasificacion', {})

        if descripcion:
            st.markdown("#### Descripción del archivo")
            st.markdown(descripcion)

        if clasificacion:
            st.dataframe(df.head(), use_container_width=True)

            st.markdown("#### Clasificación detectada")
            st.table(
                pd.DataFrame.from_dict(clasificacion, orient='index', columns=['Tipo'])
                .rename_axis('Columna')
                .reset_index()
            )

            # Aplicar tipos antes del perfilado
            df_clasificado = df.copy()
            df_clasificado.columns = df_clasificado.columns.str.strip()

            for columna, tipo in clasificacion.items():
                col = columna.strip()
                if col in df_clasificado.columns:
                    try:
                        if tipo in ["Numérica", "Booleana"]:
                            df_clasificado[col] = pd.to_numeric(df_clasificado[col], errors='coerce')
                        elif tipo == "Temporal":
                            df_clasificado[col] = pd.to_datetime(df_clasificado[col], errors='coerce')
                        elif tipo in ["Categórica"]:
                            df_clasificado[col] = df_clasificado[col].astype('category')
                        else:
                            df_clasificado[col] = df_clasificado[col].astype(str)
                    except Exception as e:
                        st.warning(f"No se pudo convertir '{col}': {e}")
                else:
                    st.warning(f"Columna '{col}' no encontrada en el DataFrame")

            with st.spinner("Generando reporte de perfilado..."):
                profile = ydata_profiling.ProfileReport(df_clasificado, minimal=False, explorative=True)
                st.components.v1.html(profile.to_html(), height=600, scrolling=True)
        else:
            st.warning(f"No se pudieron clasificar las variables de {nombre}")

        if nombre == list(st.session_state['archivos'].keys())[-1]:
            st.session_state['pasar_a_limpieza'] = True
else:
    st.session_state['pasar_a_limpieza'] = False

# Limpieza de los datos
if st.session_state.get('pasar_a_limpieza') == True:
    st.subheader("Limpieza básica de Datos")
    st.markdown("En esta sección, puedes realizar limpieza de datos, como eliminar duplicados, manejar valores faltantes, etc.")

    # Aquí podrías agregar más funcionalidades de limpieza según sea necesario
    st.info("Funcionalidades de limpieza aún no implementadas. Vuelve pronto.")