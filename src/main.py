import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import numpy as np

from modules.load import filtrar_duplicados, leer_archivos
from modules.preprocessing import clasificar_variables, describir_archivo, generar_perfilado, mostrar_outliers
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.preprocessing import mostrar_outliers


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
st.title("Análisis Automático de Datos")
st.subheader("Acá va una descripción breve del proyecto o de la aplicación.")

# --- API Key y modelo ---
model = cargar_api_gemini()
st.session_state['model'] = model
# --- SECCIÓN DE CARGA DE ARCHIVOS ---

_ = """ # --- Carga de archivos ---
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
                "A continuación, seleccione los archivos que desea incluir en el análisis conjunto."

                "**Recomendación:** Seleccione archivos que estén relacionados entre sí, es decir, que compartan claves comunes, tengan la misma estructura o puedan combinarse sin problemas."

                "**Nota:** Los archivos que no sean seleccionados serán excluidos del análisis."
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

# Perfilado de los datos
if st.session_state.get('pasar_a_perfilado') == True:
    st.title("Análisis Exploratorio de Datos")

    # Clasificación de variables
    for nombre, datos in st.session_state['archivos'].items():
        df = datos['dataframe']
        st.markdown(f"## {nombre}")

        if datos.get('descripcion') is None:
            with st.spinner("Describiendo archivo..."):
                try:
                    descripcion = describir_archivo(df, model)
                    # descripcion = descripcion_safe  # Usar descripción segura para pruebas
                    st.session_state['archivos'][nombre]['descripcion'] = descripcion
                except Exception as e:
                    st.error(f"Error al describir el archivo de {nombre}: {e}")
                    st.session_state['archivos'][nombre]['descripcion'] = {}

        descripcion = st.session_state['archivos'][nombre].get('descripcion', {})

        if datos.get('clasificacion') is None:
            with st.spinner("Clasificando variables..."):
                try:
                    clasificacion = clasificar_variables(df, model)
                    # clasificacion = clasificacion_safe  # Usar clasificación segura para pruebas
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

            # Generar perfilado usando la función
             with st.spinner("Generando reporte de perfilado..."):
                profile = generar_perfilado(df, clasificacion)
                if profile:
                    st.markdown("#### Perfilado Inicial de Datos")
                    st.components.v1.html(profile.to_html(), height=600, scrolling=True)
                    # correlaciones = profile.get_description().correlations # así se accede a las correlaciones
                else:
                    st.warning(f"No se pudo generar el perfilado para {nombre}")
        else:
            st.warning(f"No se pudieron clasificar las variables de {nombre}")

        if nombre == list(st.session_state['archivos'].keys())[-1]:
            st.session_state['pasar_a_limpieza'] = True

# Limpieza de los datos
if st.session_state.get('pasar_a_limpieza') == True:
    st.title("Limpieza de Datos")

    # Registro de operaciones realizadas
    operaciones_realizadas = []

    for nombre, datos in st.session_state['archivos'].items():
        df = datos['dataframe']
        clasificacion = datos.get('clasificacion', {})

        if not clasificacion:
            st.warning(f"No se puede limpiar el archivo '{nombre}' porque no tiene clasificación.")
            continue

        # Limpia los nombres de las columnas
        df.columns = df.columns.str.strip()

        for columna, tipo in clasificacion.items():
            col = columna.strip()
            if col in df.columns:
                if df[col].isnull().sum() > 0:
                    if tipo == "Numérica":
                        # Reemplazar NaN con el promedio
                        df[col] = pd.to_numeric(df[col], errors='coerce')  # Asegurar que sea numérico
                        mean_value = df[col].mean()
                        df[col] = df[col].fillna(mean_value)
                        operaciones_realizadas.append({
                            'Archivo': nombre,
                            'Columna': col,
                            'Tipo': tipo,
                            'Operación': f"Reemplazado NaN con promedio ({mean_value:.2f})"
                        })
                    elif tipo in ["Categórica", "Booleana", "Temporal"]:
                        # Reemplazar NaN con el valor más común
                        mode_series = df[col].mode()
                        mode_value = mode_series[0] if not mode_series.empty else None
                        if mode_value is not None:
                            df[col] = df[col].fillna(mode_value)
                            operaciones_realizadas.append({
                                'Archivo': nombre,
                                'Columna': col,
                                'Tipo': tipo,
                                'Operación': f"Reemplazado NaN con valor más común ({mode_value})"
                            })
                        else:
                            operaciones_realizadas.append({
                                'Archivo': nombre,
                                'Columna': col,
                                'Tipo': tipo,
                                'Operación': "No se pudo determinar el valor más común para reemplazar NaN"
                            })
        st.session_state['archivos'][nombre]['dataframe_limpio'] = df

    # Mostrar resumen de operaciones realizadas
    st.markdown("### Resumen de Operaciones Realizadas")
    if operaciones_realizadas:
        operaciones_df = pd.DataFrame(operaciones_realizadas)
        st.dataframe(operaciones_df, use_container_width=True)
    else:
        st.info("No se realizaron operaciones durante la limpieza.")
    st.session_state['pasar_a_analisis'] = True
else:
    st.session_state['pasar_a_analisis'] = False



# --- Análisis de los datos ---
if st.session_state.get('pasar_a_analisis') == True:
    st.title("Preparación para IA")
    st.markdown("En esta sección, se realiza la codificación de las columnas categóricas y la estandarización de las columnas numéricas para análisis avanzado.")

    for nombre, datos in st.session_state['archivos'].items():
        df = datos['dataframe_limpio'] if 'dataframe_limpio' in datos else datos['dataframe']
        clasificacion = datos.get('clasificacion', {})

        if not clasificacion:
            st.warning(f"No se puede analizar el archivo '{nombre}' porque no tiene clasificación.")
            continue

        st.markdown(f"### {nombre}")

        # Identificar columnas categóricas y numéricas
        cat_attribs = [col.strip() for col, tipo in clasificacion.items() if tipo == "Categórica" and col.strip() in df.columns]
        num_attribs = [col.strip() for col, tipo in clasificacion.items() if tipo == "Numérica" and col.strip() in df.columns]

        if not cat_attribs and not num_attribs:
            st.info(f"No se encontraron columnas categóricas o numéricas en el archivo '{nombre}'.")
            continue

        # Crear el OneHotEncoder para columnas categóricas
        encoder = OneHotEncoder(handle_unknown='ignore')

        # Aplicar el encoder a las columnas categóricas
        try:
            encoded_cats = encoder.fit_transform(df[cat_attribs])

            # Crear un DataFrame con las columnas codificadas
            encoded_cats_df = pd.DataFrame(
                encoded_cats.toarray(),
                columns=encoder.get_feature_names_out(cat_attribs),
                index=df.index
            )

            # Concatenar las columnas codificadas con el DataFrame original
            df_encoded = pd.concat([df.drop(columns=cat_attribs), encoded_cats_df], axis=1)

            # Combinar columnas numéricas originales con las columnas codificadas
            all_num_attribs = num_attribs + list(encoded_cats_df.columns)

            # Aplicar el StandardScaler a todas las columnas numéricas y codificadas
            scaler = StandardScaler()
            scaled_nums = scaler.fit_transform(df_encoded[all_num_attribs])

            # Crear un DataFrame con las columnas estandarizadas
            scaled_nums_df = pd.DataFrame(
                scaled_nums,
                columns=all_num_attribs,
                index=df_encoded.index
            )

            # Reemplazar las columnas originales con las estandarizadas
            df_final = df_encoded.copy()
            df_final[all_num_attribs] = scaled_nums_df

            # Mostrar vista previa del DataFrame final
            st.markdown("#### Vista Previa del DataFrame Final")
            st.dataframe(df_final.head(), use_container_width=True)

            # Mostrar gráfico de outliers
            st.markdown("#### Gráfico de Outliers")
            mostrar_outliers(df_final, num_attribs) 

            # 2. KMeans
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(df_final[all_num_attribs])
            # 3. Añadir al dataframe
            df_final["cluster"] = clusters

            st.markdown("#### Tabla de Datos Finales con Clusters")
            st.dataframe(df_final, use_container_width=True)

            # Media por cluster
            insight_table = df_final.groupby("cluster").mean(numeric_only=True)
            st.subheader("Promedios por grupo")
            st.dataframe(insight_table)

            # Visualizar agrupaciones con PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(df_final[all_num_attribs])
            df_final["PC1"] = X_pca[:,0]
            df_final["PC2"] = X_pca[:,1]

            sns.scatterplot(data=df_final, x="PC1", y="PC2", hue="cluster", palette="Set2")
            plt.title("Clusters visualizados con PCA")
            st.pyplot(plt.gcf())

            # Guardar el DataFrame final en session_state
            st.session_state['archivos'][nombre]['dataframe_final'] = df_final

            st.success(f"Codificación y estandarización completadas para el archivo: {nombre}")
        except Exception as e:
            st.error(f"Error al procesar el archivo '{nombre}': {e}")

 """