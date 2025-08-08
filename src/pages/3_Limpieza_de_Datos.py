import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from modules.preprocessing import detectar_outliers_zscore, detectar_outliers_iqr, detectar_outliers_isolation_forest, encoding_categorias, mostrar_resumen_outliers

if st.session_state.get('pasar_a_limpieza') == True:
    st.title("Preprocesamiento de Datos")

    for nombre, datos in st.session_state['archivos'].items():
        # Copia para trabajar sin modificar el original
        df_original = datos['dataframe'].copy()
        clasificacion = datos.get('clasificacion', {})

        # Limpia nombres de columnas
        df_original.columns = df_original.columns.str.strip()

        # Inicializar registro de operaciones
        if 'operaciones_realizadas' not in datos:
            datos['operaciones_realizadas'] = []

        # ===== IMPUTACIÓN DE NULOS =====
        for columna, tipo in clasificacion.items():
            col = columna.strip()
            if col in df_original.columns and df_original[col].isnull().sum() > 0:
                if tipo == "Numérica":
                    df_original[col] = pd.to_numeric(df_original[col], errors='coerce')
                    mean_value = df_original[col].mean()
                    df_original[col] = df_original[col].fillna(mean_value)
                    datos['operaciones_realizadas'].append({
                        'Archivo': nombre, 'Columna': col, 'Tipo': tipo,
                        'Operación': f"Reemplazado NaN con promedio ({mean_value:.2f})"
                    })
                elif tipo in ["Categórica", "Booleana", "Temporal"]:
                    mode_series = df_original[col].mode()
                    mode_value = mode_series[0] if not mode_series.empty else None
                    if mode_value is not None:
                        df_original[col] = df_original[col].fillna(mode_value)
                        datos['operaciones_realizadas'].append({
                            'Archivo': nombre, 'Columna': col, 'Tipo': tipo,
                            'Operación': f"Reemplazado NaN con valor más común ({mode_value})"
                        })
                    else:
                        datos['operaciones_realizadas'].append({
                            'Archivo': nombre, 'Columna': col, 'Tipo': tipo,
                            'Operación': "No se pudo determinar el valor más común"
                        })

        # Mostrar operaciones
        st.markdown(f"### Archivo: {nombre}")
        st.markdown("#### Imputación de Nulos")
        if datos['operaciones_realizadas']:
            st.dataframe(pd.DataFrame(datos['operaciones_realizadas']), use_container_width=True)
        else:
            st.info(f"No se realizaron operaciones durante la limpieza para el archivo '{nombre}'.")

        # Identificar columnas
        cat_attribs = [col.strip() for col, tipo in clasificacion.items() if tipo == "Categórica" and col.strip() in df_original.columns]
        num_attribs = [col.strip() for col, tipo in clasificacion.items() if tipo == "Numérica" and col.strip() in df_original.columns]

        # ===== DETECCIÓN DE OUTLIERS =====
        st.markdown("#### Detección de Outliers")
        metodo = st.selectbox("Método de detección", ["Z-Score", "IQR", "Isolation Forest"], key=f"metodo_{nombre}")

        if metodo == "Z-Score":
            outliers = detectar_outliers_zscore(df_original, num_attribs)
        elif metodo == "IQR":
            outliers = detectar_outliers_iqr(df_original, num_attribs)
        else:
            outliers = detectar_outliers_isolation_forest(df_original, num_attribs)

        mostrar_resumen_outliers(df_original, num_attribs, outliers, metodo)

        accion = st.radio("¿Qué hacer con los outliers?", [
            "Conservar todos los datos",
            "Excluir outliers del análisis",
            "Etiquetarlos en una nueva columna"
        ], key=f"accion_{nombre}")

        if accion == "Excluir outliers del análisis":
            df_limpio = df_original[~outliers].copy()
            st.success(f"Eliminados {outliers.sum()} outliers")
        elif accion == "Etiquetarlos en una nueva columna":
            df_limpio = df_original.copy()
            df_limpio['es_outlier'] = outliers
            st.success("Columna 'es_outlier' añadida")
        else:
            df_limpio = df_original.copy()

        # ===== CODIFICACIÓN Y ESTANDARIZACIÓN =====
        st.markdown("#### Codificación y Estandarización de Datos")
        if cat_attribs:
            df_procesado, columnas_finales = encoding_categorias(df_limpio, cat_attribs, num_attribs, nombre)
            st.success(f"Codificadas {len(cat_attribs)} variable(s) categórica(s)")
        else:
            scaler = StandardScaler()
            df_procesado = df_limpio.copy()
            df_procesado[num_attribs] = scaler.fit_transform(df_limpio[num_attribs])
            columnas_finales = num_attribs

        st.dataframe(df_procesado.head(), use_container_width=True)

        # Guardar en session_state ambas versiones
        datos['dataframe_limpio'] = df_limpio      # con imputación y outliers procesados
        datos['dataframe_final'] = df_procesado # con codificación/estandarización final

    st.session_state['pasar_a_analisis'] = True
else:
    st.warning("Primero completa la sección de Análisis Exploratorio de Datos.")
    st.session_state['pasar_a_analisis'] = False
