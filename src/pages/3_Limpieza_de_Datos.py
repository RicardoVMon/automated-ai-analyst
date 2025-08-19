import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from modules.preprocessing import detectar_outliers_zscore, detectar_outliers_iqr, detectar_outliers_isolation_forest, encoding_categorias, mostrar_resumen_outliers

if st.session_state.get('pasar_a_limpieza') == True:
    st.title("Preprocesamiento de Datos")

    archivos = list(st.session_state['archivos'].items())
    nombres = [nombre for nombre, _ in archivos]
    tabs = st.tabs(nombres)

    for i, (nombre, datos) in enumerate(archivos):
        with tabs[i]:
            df_original = datos['dataframe'].copy()
            clasificacion = datos.get('clasificacion', {})
            df_original.columns = df_original.columns.str.strip()
            if 'operaciones_realizadas' not in datos:
                datos['operaciones_realizadas'] = []

            etapa_tabs = st.tabs([
                "Imputación de Nulos",
                "Borrado de Alta Cardinalidad",
                "Detección de Outliers",
                "Codificación y Estandarización"
            ])

            # ===== IMPUTACIÓN DE NULOS =====
            with etapa_tabs[0]:
                if 'df_imputado' not in datos:
                    df_imputado = df_original.copy()
                    for columna, tipo in clasificacion.items():
                        col = columna.strip()
                        if col in df_imputado.columns and df_imputado[col].isnull().sum() > 0:
                            if tipo == "Numérica":
                                df_imputado[col] = pd.to_numeric(df_imputado[col], errors='coerce')
                                mean_value = df_imputado[col].mean()
                                df_imputado[col] = df_imputado[col].fillna(mean_value)
                                datos['operaciones_realizadas'].append({
                                    'Archivo': nombre, 'Columna': col, 'Tipo': tipo,
                                    'Operación': f"Reemplazado NaN con promedio ({mean_value:.2f})"
                                })
                            elif tipo in ["Categórica", "Booleana", "Temporal"]:
                                mode_series = df_imputado[col].mode()
                                mode_value = mode_series[0] if not mode_series.empty else None
                                if mode_value is not None:
                                    df_imputado[col] = df_imputado[col].fillna(mode_value)
                                    datos['operaciones_realizadas'].append({
                                        'Archivo': nombre, 'Columna': col, 'Tipo': tipo,
                                        'Operación': f"Reemplazado NaN con valor más común ({mode_value})"
                                    })
                                else:
                                    datos['operaciones_realizadas'].append({
                                        'Archivo': nombre, 'Columna': col, 'Tipo': tipo,
                                        'Operación': "No se pudo determinar el valor más común"
                                    })
                    datos['df_imputado'] = df_imputado
                else:
                    df_imputado = datos['df_imputado']

                st.markdown("#### Imputación de Nulos")
                if datos['operaciones_realizadas']:
                    st.dataframe(pd.DataFrame(datos['operaciones_realizadas']), use_container_width=True)
                else:
                    st.info(f"No fue necesario realizar operaciones durante la limpieza para el archivo '{nombre}'.")

            # Identificar columnas
            cat_attribs = [col.strip() for col, tipo in clasificacion.items() if tipo == "Categórica" and col.strip() in df_imputado.columns]
            num_attribs = [col.strip() for col, tipo in clasificacion.items() if tipo == "Numérica" and col.strip() in df_imputado.columns]

            # ===== BORRADO DE ALTA CARDINALIDAD =====
            with etapa_tabs[1]:
                df_card = datos.get('df_card', df_imputado.copy())
                st.markdown("#### Borrado de Columnas de Alta Cardinalidad")
                umbral = st.number_input(
                    "Define el umbral de valores únicos para considerar alta cardinalidad:",
                    min_value=2, max_value=10000, value=100, step=1, key=f"umbral_cardinalidad_{nombre}"
                )
                columnas_alta_cardinalidad = [(col, df_card[col].nunique()) for col in df_card.columns if df_card[col].nunique() > umbral]
                if columnas_alta_cardinalidad:
                    st.warning(f"Las siguientes columnas tienen alta cardinalidad (> {umbral} valores únicos):")
                    df_cardinalidad = pd.DataFrame(columnas_alta_cardinalidad, columns=["Columna", "Valores Únicos"])
                    st.dataframe(df_cardinalidad, use_container_width=True)
                    nombres_columnas = [col for col, _ in columnas_alta_cardinalidad]
                    columnas_a_eliminar = st.multiselect(
                        "Selecciona las columnas que deseas eliminar por alta cardinalidad:",
                        nombres_columnas,
                        placeholder="Selecciona una o varias columnas",
                        key=f"accion_cardinalidad_{nombre}"
                    )
                    if st.button("Eliminar columnas seleccionadas", key=f"eliminar_cardinalidad_{nombre}"):
                        df_card = df_card.drop(columns=columnas_a_eliminar)
                        datos['df_card'] = df_card
                        st.success(f"Columnas eliminadas: {columnas_a_eliminar}")
                    else:
                        datos['df_card'] = df_card
                else:
                    st.info("No se detectaron columnas con alta cardinalidad.")
                    datos['df_card'] = df_card

            # Actualizar atributos categóricos y numéricos tras borrado
            cat_attribs_card = [col for col in cat_attribs if col in datos['df_card'].columns]
            num_attribs_card = [col for col in num_attribs if col in datos['df_card'].columns]

             # ===== DETECCIÓN DE OUTLIERS =====
            with etapa_tabs[2]:
                # SIEMPRE partir del resultado de la etapa anterior
                df_outliers = datos['df_card'].copy()
                st.markdown("#### Detección de Outliers")
                metodo = st.selectbox("Método de detección", ["Z-Score", "IQR", "Isolation Forest"], key=f"metodo_{nombre}")
                if metodo == "Z-Score":
                    outliers = detectar_outliers_zscore(df_outliers, num_attribs_card)
                elif metodo == "IQR":
                    outliers = detectar_outliers_iqr(df_outliers, num_attribs_card)
                else:
                    outliers = detectar_outliers_isolation_forest(df_outliers, num_attribs_card)
                mostrar_resumen_outliers(df_outliers, num_attribs_card, outliers, metodo)
                accion = st.radio("¿Qué hacer con los outliers?", [
                    "Conservar todos los datos",
                    "Excluir outliers del análisis"
                ], key=f"accion_outliers_{nombre}")
                if accion == "Excluir outliers del análisis":
                    df_limpio = df_outliers[~outliers].copy()
                    st.success(f"Eliminados {outliers.sum()} outliers")
                else:
                    df_limpio = df_outliers.copy()
                datos['df_limpio_outliers'] = df_limpio

            # Actualizar atributos categóricos y numéricos tras outliers
            cat_attribs_limpio = [col for col in cat_attribs_card if col in datos['df_limpio_outliers'].columns]
            num_attribs_limpio = [col for col in num_attribs_card if col in datos['df_limpio_outliers'].columns]

            # ===== CODIFICACIÓN Y ESTANDARIZACIÓN =====
            with etapa_tabs[3]:
                st.markdown("#### Codificación y Estandarización de Datos")
                df_limpio = datos['df_limpio_outliers']
                if cat_attribs_limpio:
                    df_procesado, columnas_finales = encoding_categorias(df_limpio, cat_attribs_limpio, num_attribs_limpio, nombre)
                    st.success(f"Codificadas {len(cat_attribs_limpio)} variable(s) categórica(s)")
                else:
                    scaler = StandardScaler()
                    df_procesado = df_limpio.copy()
                    df_procesado[num_attribs_limpio] = scaler.fit_transform(df_limpio[num_attribs_limpio])
                    columnas_finales = num_attribs_limpio
                st.dataframe(df_procesado.head(), use_container_width=True)
                datos['dataframe_final'] = df_procesado

            # Guardar en session_state ambas versiones
            datos['dataframe_limpio'] = datos['df_limpio_outliers']  # con imputación y outliers procesados

    st.session_state['pasar_a_analisis'] = True
else:
    st.warning("Primero completa la sección de Análisis Exploratorio de Datos.")
    st.session_state['pasar_a_analisis'] = False