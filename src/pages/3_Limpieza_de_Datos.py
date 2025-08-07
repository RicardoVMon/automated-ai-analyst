import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from modules.preprocessing import detectar_outliers_zscore, detectar_outliers_iqr, detectar_outliers_isolation_forest, encoding_categorias, mostrar_resumen_outliers

# Limpieza de los datos
if st.session_state.get('pasar_a_limpieza') == True:
    st.title("Preprocesamiento de Datos")

    for nombre, datos in st.session_state['archivos'].items():
        df = datos['dataframe']
        clasificacion = datos.get('clasificacion', {})

        # Limpia los nombres de las columnas
        df.columns = df.columns.str.strip()

        # Inicializar el registro de operaciones para este archivo si no existe
        if 'operaciones_realizadas' not in st.session_state['archivos'][nombre]:
            st.session_state['archivos'][nombre]['operaciones_realizadas'] = []

        for columna, tipo in clasificacion.items():
            col = columna.strip()
            if col in df.columns:
                if df[col].isnull().sum() > 0:
                    if tipo == "Numérica":
                        # Reemplazar NaN con el promedio
                        df[col] = pd.to_numeric(df[col], errors='coerce')  # Asegurar que sea numérico
                        mean_value = df[col].mean()
                        df[col] = df[col].fillna(mean_value)
                        st.session_state['archivos'][nombre]['operaciones_realizadas'].append({
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
                            st.session_state['archivos'][nombre]['operaciones_realizadas'].append({
                                'Archivo': nombre,
                                'Columna': col,
                                'Tipo': tipo,
                                'Operación': f"Reemplazado NaN con valor más común ({mode_value})"
                            })
                        else:
                            st.session_state['archivos'][nombre]['operaciones_realizadas'].append({
                                'Archivo': nombre,
                                'Columna': col,
                                'Tipo': tipo,
                                'Operación': "No se pudo determinar el valor más común para reemplazar NaN"
                            })

        st.markdown(f"### Archivo: {nombre}")
        st.markdown("#### Imputación de Nulos")
        operaciones = datos.get('operaciones_realizadas', [])
        if operaciones:
            operaciones_df = pd.DataFrame(operaciones)
            st.dataframe(operaciones_df, use_container_width=True)
        else:
            st.info(f"No se realizaron operaciones durante la limpieza para el archivo '{nombre}'.")

        # Identificar columnas categóricas y numéricas
        cat_attribs = [col.strip() for col, tipo in clasificacion.items() if tipo == "Categórica" and col.strip() in df.columns]
        num_attribs = [col.strip() for col, tipo in clasificacion.items() if tipo == "Numérica" and col.strip() in df.columns]

        # Codificación y estandarización
        st.markdown("#### Codificación y Estandarización de Datos")
        if cat_attribs:
            df_procesado, columnas_finales = encoding_categorias(df, cat_attribs, num_attribs, nombre)
            st.success(f"Codificadas {len(cat_attribs)} variable(s) categórica(s)")
        else:
            # Estandarizar solo las numéricas si no hay categóricas CAMBIAR
            scaler = StandardScaler()
            df_procesado = df.copy()
            df_procesado[num_attribs] = scaler.fit_transform(df[num_attribs])
            columnas_finales = num_attribs
            
        st.dataframe(df_procesado.head(), use_container_width=True)

        # Detección de outliers
        st.markdown("#### Detección de Outliers")
        metodo = st.selectbox("Método de detección", ["Z-Score", "IQR", "Isolation Forest"], key=f"metodo_{nombre}")

        if metodo == "Z-Score":
            outliers = detectar_outliers_zscore(df_procesado, num_attribs)
        elif metodo == "IQR":
            outliers = detectar_outliers_iqr(df_procesado, num_attribs)
        else:
            outliers = detectar_outliers_isolation_forest(df_procesado, num_attribs)

        mostrar_resumen_outliers(df_procesado, num_attribs, outliers, metodo)

        accion = st.radio("¿Qué hacer con los outliers?", [
            "Conservar todos los datos",
            "Excluir outliers del análisis",
            "Etiquetarlos en una nueva columna"
        ], key=f"accion_{nombre}")

        if accion == "Excluir outliers del análisis":
            df_final = df_procesado[~outliers].copy()
            st.success(f"Eliminados {outliers.sum()} outliers")
        elif accion == "Etiquetarlos en una nueva columna":
            df_final = df_procesado.copy()
            df_final['es_outlier'] = outliers
            st.success("Columna 'es_outlier' añadida")
        else:
            df_final = df_procesado.copy()

        st.session_state['archivos'][nombre]['dataframe_final'] = df_final
    st.session_state['pasar_a_analisis'] = True
else:
    st.warning("Primero completa la sección de Análisis Exploratorio de Datos.")
    st.session_state['pasar_a_analisis'] = False