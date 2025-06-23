import streamlit as st
import pandas as pd

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

# --- Función para clasificar variables ---
def clasificar_variables(df):
    variables = {
        "Numericas": [],
        "Categoricas": [],
        "Temporales": [],
        "Booleanas": [],
        "Desconocidas": []
    }

    for col in df.columns:
        serie = df[col]
        dtype = serie.dtype

        if pd.api.types.is_bool_dtype(dtype):
            variables["Booleanas"].append(col)

        elif pd.api.types.is_numeric_dtype(dtype):
            valores_unicos = serie.dropna().unique()
            if len(valores_unicos) == 2 and set(valores_unicos).issubset({0, 1}):
                variables["Booleanas"].append(col)
            else:
                variables["Numericas"].append(col)

        elif pd.api.types.is_datetime64_any_dtype(dtype):
            variables["Temporales"].append(col)

        elif pd.api.types.is_object_dtype(dtype):
            try:
                converted = pd.to_datetime(serie, errors="coerce")
                if converted.notna().mean() > 0.9:
                    variables["Temporales"].append(col)
                    continue
            except Exception:
                pass

            valores_unicos = serie.dropna().astype(str).str.lower().unique()
            if len(valores_unicos) == 2 and set(valores_unicos).issubset({"yes", "no", "true", "false"}):
                variables["Booleanas"].append(col)
            else:
                variables["Categoricas"].append(col)

        elif pd.api.types.is_categorical_dtype(dtype):
            variables["Categoricas"].append(col)

        else:
            variables["Desconocidas"].append(col)

    return variables

# --- Interfaz de usuario ---
st.title("Análisis de Datos Inicial")

uploaded_file = st.file_uploader("Carga tu archivo CSV o Excel", type=["csv", "xlsx"])

if uploaded_file:
    df = leer_archivo(uploaded_file)

    if df is not None:
        st.success("Archivo cargado correctamente")

        st.subheader("Vista previa de los datos")
        st.dataframe(df.head())

        st.subheader("Clasificación automática de variables")
        tipos = clasificar_variables(df)

         # Mostrar clasificación como tabla
        clasificacion = []
        for tipo, columnas in tipos.items():
            for col in columnas:
                clasificacion.append({"Columna": col, "Tipo de variable": tipo})

        df_clasificacion = pd.DataFrame(clasificacion)

        st.dataframe(df_clasificacion, use_container_width=True)
else:
    st.info("Esperando archivo...")