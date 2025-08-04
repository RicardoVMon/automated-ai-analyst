from modules.prompts import prompt_clasificacion_variables, prompt_describir_archivo
import ydata_profiling, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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

def describir_archivo(df, model):
    prompt = prompt_describir_archivo(df)
    response = model.generate_content(prompt)
    texto = response.text.strip()
    return texto

def generar_perfilado(df, clasificacion):
    """
    Genera un perfilado de datos basado en la clasificación de variables.
    
    Args:
        df (pd.DataFrame): DataFrame original.
        clasificacion (dict): Diccionario con la clasificación de las variables.
    
    Returns:
        ydata_profiling.ProfileReport: Reporte de perfilado generado.
    """
    try:
        # Aplicar tipos antes del perfilado
        df_clasificado = df.copy()
        df_clasificado.columns = df_clasificado.columns.str.strip()

        for columna, tipo in clasificacion.items():
            col = columna.strip()
            if col in df_clasificado.columns:
                if tipo in ["Numérica", "Booleana"]:
                    df_clasificado[col] = pd.to_numeric(df_clasificado[col], errors='coerce')
                elif tipo == "Temporal":
                    df_clasificado[col] = pd.to_datetime(df_clasificado[col], errors='coerce')
                elif tipo in ["Categórica"]:
                    df_clasificado[col] = df_clasificado[col].astype('category')
                else:
                    df_clasificado[col] = df_clasificado[col].astype(str)

        # Generar reporte de perfilado
        profile = ydata_profiling.ProfileReport(df_clasificado, minimal=False, explorative=True)
        return profile

    except Exception as e:
        return None
    
def mostrar_outliers(df, columnas_num):
    """
    Genera un gráfico de boxplots para identificar outliers en las columnas numéricas.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        columnas_num (list): Lista de columnas numéricas a analizar.

    Returns:
        None: Muestra el gráfico en Streamlit.
    """
    if not columnas_num:
        st.warning("No se encontraron columnas numéricas para analizar outliers.")
        return

    # Configurar el tamaño del gráfico
    plt.figure(figsize=(12, 6))

    # Crear un boxplot para cada columna numérica
    sns.boxplot(data=df[columnas_num], orient="h", palette="Set2")

    # Configurar el título y las etiquetas
    plt.title("Detección de Outliers en Columnas Numéricas", fontsize=16)
    plt.xlabel("Valores", fontsize=12)
    plt.ylabel("Columnas", fontsize=12)

    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)