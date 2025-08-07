from modules.prompts import prompt_clasificacion_variables, prompt_describir_archivo
import ydata_profiling, pandas as pd
import streamlit as st
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    
# --- Función 1: Detección por Z-Score ---
def detectar_outliers_zscore(df, columnas, threshold=3):
    z_scores = df[columnas].apply(zscore)
    outliers = (z_scores.abs() > threshold).any(axis=1)
    return outliers

# --- Función 2: Detección por IQR ---
def detectar_outliers_iqr(df, columnas):
    outliers = pd.Series(False, index=df.index)
    for col in columnas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        is_outlier = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
        outliers |= is_outlier
    return outliers

# --- Función 3: Detección por Isolation Forest ---
def detectar_outliers_isolation_forest(df, columnas):
    modelo = IsolationForest(contamination=0.05, random_state=42)
    pred = modelo.fit_predict(df[columnas])
    outliers = pred == -1
    return pd.Series(outliers, index=df.index)

def encoding_categorias(df, cat_attribs, num_attribs, nombre):
    """Codifica las columnas categóricas utilizando One-Hot Encoding."""
    encoder = OneHotEncoder(handle_unknown='ignore')
    
    try:
        encoded_cats = encoder.fit_transform(df[cat_attribs])
        encoded_cats_df = pd.DataFrame(
            encoded_cats.toarray(),
            columns=encoder.get_feature_names_out(cat_attribs),
            index=df.index
        )
        
        df_encoded = pd.concat([df.drop(columns=cat_attribs), encoded_cats_df], axis=1)
        all_num_attribs = num_attribs + list(encoded_cats_df.columns)
        
        scaler = StandardScaler()
        scaled_nums = scaler.fit_transform(df_encoded[all_num_attribs])
        
        scaled_nums_df = pd.DataFrame(
            scaled_nums,
            columns=all_num_attribs,
            index=df_encoded.index
        )
        
        df_final = df_encoded.copy()
        df_final[all_num_attribs] = scaled_nums_df
        
        st.session_state['archivos'][nombre]['dataframe_final'] = df_final
        return df_final, all_num_attribs
        
    except Exception as e:
        st.error(f"Error al procesar el archivo '{nombre}': {e}")
        return df, num_attribs

# --- Visualización general con tabs por variable ---
def mostrar_resumen_outliers(df, columnas, outliers, metodo):
    st.markdown(f"#### Resultado de detección de outliers ({metodo})")
    st.write(f"Se detectaron {outliers.sum()} outliers sobre {len(df)} registros.")
    st.dataframe(df[outliers].head(), use_container_width=True)

    st.markdown("#### Distribución de variables numéricas")
    tabs = st.tabs(columnas)
    for i, col in enumerate(columnas):
        with tabs[i]:
            # Crear gráfico interactivo con Plotly
            fig = px.box(
                df, 
                y=col, 
                title=f"Boxplot de {col} (Outliers resaltados)",
                points="outliers"  # Mostrar solo los outliers
            )
            st.plotly_chart(fig, use_container_width=True)