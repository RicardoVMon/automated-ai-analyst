from modules.prompts import prompt_clasificacion_variables, prompt_describir_archivo
import ydata_profiling, pandas as pd
import streamlit as st
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Función para clasificar variables con un modelo de lenguaje ---
def clasificar_variables(df, model):
    """
    Clasifica automáticamente las variables de un DataFrame utilizando un modelo de lenguaje.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos a analizar.
        model: Modelo de lenguaje capaz de generar texto (ejemplo: LLM).

    Retorna:
        dict: Diccionario con columnas como claves y su tipo inferido como valor.
    """
    prompt = prompt_clasificacion_variables(df) # Genera un prompt con la info del DataFrame
    response = model.generate_content(prompt) # Envía el prompt al modelo y obtiene respuesta
    texto = response.text.strip() # Limpia espacios extra de la respuesta

    clasificacion = {} # Diccionario para guardar resultados
    for linea in texto.splitlines():  # Itera sobre cada línea de la respuesta
        if ':' in linea:  # Si la línea tiene el formato "columna: tipo"
            col, tipo = linea.split(':', 1) # Separa en nombre de columna y tipo
            clasificacion[col.strip()] = tipo.strip()  # Agrega al diccionario sin espacios
    return clasificacion # Retorna el diccionario final


# --- Función para describir un archivo ---
def describir_archivo(df, model):
    """
    Genera una descripción textual de un archivo (DataFrame) utilizando un modelo de lenguaje.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos a describir.
        model: Modelo de lenguaje capaz de generar texto.

    Retorna:
        str: Descripción generada en formato texto.
    """
    prompt = prompt_describir_archivo(df)         # Genera un prompt con la info del DataFrame
    response = model.generate_content(prompt)     # Envía el prompt al modelo
    texto = response.text.strip()                 # Limpia espacios de la respuesta
    return texto                                  # Retorna la descripción


# --- Función para generar un perfilado automático ---
def generar_perfilado(df, clasificacion):
    """
    Genera un perfilado de datos basado en la clasificación de variables.

    Parámetros:
        df (pd.DataFrame): DataFrame original.
        clasificacion (dict): Diccionario con clasificación de variables.

    Retorna:
        ydata_profiling.ProfileReport | None: Reporte de perfilado o None si falla.
    """
    try:
        df_clasificado = df.copy()                          # Copia del DataFrame original
        df_clasificado.columns = df_clasificado.columns.str.strip()  # Limpia espacios en nombres

        for columna, tipo in clasificacion.items():          # Itera sobre la clasificación
            col = columna.strip()                            # Limpia espacios en el nombre
            if col in df_clasificado.columns:                # Solo si existe la columna
                if tipo in ["Numérica", "Booleana"]:         # Si es numérica o booleana
                    df_clasificado[col] = pd.to_numeric(df_clasificado[col], errors='coerce')
                elif tipo == "Temporal":                     # Si es de tiempo
                    df_clasificado[col] = pd.to_datetime(df_clasificado[col], errors='coerce')
                elif tipo in ["Categórica"]:                 # Si es categórica
                    df_clasificado[col] = df_clasificado[col].astype('category')
                else:                                        # Si no se reconoce, la pasa a string
                    df_clasificado[col] = df_clasificado[col].astype(str)

        profile = ydata_profiling.ProfileReport( # Genera reporte con ydata_profiling
            df_clasificado, minimal=False, explorative=True
        )
        return profile  # Retorna el reporte

    except Exception as e:  # Si ocurre error
        return None  # Retorna None


# --- Función 1: Detección de outliers por Z-Score ---
def detectar_outliers_zscore(df, columnas, threshold=3):
    """
    Detecta outliers en columnas numéricas usando Z-Score.

    Parámetros:
        df (pd.DataFrame): DataFrame a analizar.
        columnas (list): Columnas numéricas a evaluar.
        threshold (float): Umbral para definir outlier (default=3).

    Retorna:
        pd.Series: Serie booleana con True en las filas outliers.
    """
    z_scores = df[columnas].apply(zscore)                   # Calcula Z-Score por columna
    outliers = (z_scores.abs() > threshold).any(axis=1)     # Marca filas que exceden el umbral
    return outliers                                         # Retorna Serie booleana


# --- Función 2: Detección de outliers por IQR ---
def detectar_outliers_iqr(df, columnas):
    """
    Detecta outliers en columnas numéricas usando el rango intercuartílico (IQR).

    Parámetros:
        df (pd.DataFrame): DataFrame a analizar.
        columnas (list): Columnas numéricas a evaluar.

    Retorna:
        pd.Series: Serie booleana con True en las filas outliers.
    """
    outliers = pd.Series(False, index=df.index)             # Inicializa todos en False
    for col in columnas:                                    # Recorre columnas
        Q1 = df[col].quantile(0.25)                         # Calcula primer cuartil
        Q3 = df[col].quantile(0.75)                         # Calcula tercer cuartil
        IQR = Q3 - Q1                                       # Calcula rango intercuartílico
        is_outlier = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)  # Define outliers
        outliers |= is_outlier                              # Actualiza outliers globales
    return outliers                                         # Retorna Serie booleana


# --- Función 3: Detección de outliers por Isolation Forest ---
def detectar_outliers_isolation_forest(df, columnas):
    """
    Detecta outliers en columnas numéricas usando Isolation Forest.

    Parámetros:
        df (pd.DataFrame): DataFrame a analizar.
        columnas (list): Columnas numéricas a evaluar.

    Retorna:
        pd.Series: Serie booleana con True en las filas outliers.
    """
    modelo = IsolationForest(contamination=0.05, random_state=42)  # Crea modelo Isolation Forest
    pred = modelo.fit_predict(df[columnas])                        # Entrena y predice
    outliers = pred == -1                                          # Marca -1 como outlier
    return pd.Series(outliers, index=df.index)                     # Retorna Serie booleana


# --- Función para codificación y escalado ---
def encoding_categorias(df, cat_attribs, num_attribs, nombre):
    """
    Codifica variables categóricas (One-Hot Encoding) y escala variables numéricas.

    Parámetros:
        df (pd.DataFrame): DataFrame original.
        cat_attribs (list): Columnas categóricas.
        num_attribs (list): Columnas numéricas.
        nombre (str): Nombre del archivo (clave en session_state).

    Retorna:
        tuple: (DataFrame transformado, lista de columnas numéricas finales).
    """
    encoder = OneHotEncoder(handle_unknown='ignore')              # Crea codificador One-Hot
    
    try:
        encoded_cats = encoder.fit_transform(df[cat_attribs])      # Codifica categorías
        encoded_cats_df = pd.DataFrame(                           # Convierte a DataFrame
            encoded_cats.toarray(),
            columns=encoder.get_feature_names_out(cat_attribs),
            index=df.index
        )
        
        df_encoded = pd.concat([df.drop(columns=cat_attribs), encoded_cats_df], axis=1)  # Une todo
        all_num_attribs = num_attribs + list(encoded_cats_df.columns)  # Nuevas columnas numéricas
        
        scaler = StandardScaler()                                 # Crea escalador
        scaled_nums = scaler.fit_transform(df_encoded[all_num_attribs])  # Escala valores
        
        scaled_nums_df = pd.DataFrame(                            # Convierte a DataFrame escalado
            scaled_nums,
            columns=all_num_attribs,
            index=df_encoded.index
        )
        
        df_final = df_encoded.copy()                              # Copia DataFrame
        df_final[all_num_attribs] = scaled_nums_df                # Reemplaza por datos escalados
        
        st.session_state['archivos'][nombre]['dataframe_final'] = df_final  # Guarda en session_state
        return df_final, all_num_attribs                          # Retorna DataFrame final y columnas
    
    except Exception as e:                                        # Si ocurre error
        st.error(f"Error al procesar el archivo '{nombre}': {e}") # Muestra error en la interfaz
        return df, num_attribs                                    # Retorna DataFrame original


# --- Función para mostrar resumen de outliers ---
def mostrar_resumen_outliers(df, columnas, outliers, metodo):
    """
    Muestra en Streamlit el resumen de outliers detectados y sus visualizaciones.

    Parámetros:
        df (pd.DataFrame): DataFrame original.
        columnas (list): Columnas numéricas analizadas.
        outliers (pd.Series): Serie booleana con True en las filas outliers.
        metodo (str): Nombre del método usado.

    Retorna:
        None. Muestra resultados en la interfaz.
    """
    st.markdown(f"#### Resultado de detección de outliers ({metodo})")  # Título con método
    st.write(f"Se detectaron {outliers.sum()} outliers sobre {len(df)} registros.")  # Conteo
    st.dataframe(df[outliers].head(), use_container_width=True)        # Muestra primeros outliers

    st.markdown("#### Distribución de variables numéricas")            # Sección de gráficos
    tabs = st.tabs(columnas)                                           # Crea pestañas por columna
    for i, col in enumerate(columnas):                                 # Itera columnas
        with tabs[i]:
            fig = px.box(                                              # Gráfico tipo boxplot
                df, 
                y=col, 
                title=f"Boxplot de {col} (Outliers resaltados)",
                points="outliers"  # Solo muestra puntos considerados outliers
            )
            st.plotly_chart(fig, use_container_width=True)             # Renderiza en Streamlit
