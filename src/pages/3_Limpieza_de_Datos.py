# =========================
# Importación de librerías y módulos
# =========================

import streamlit as st  # Importa Streamlit para la interfaz web
import pandas as pd  # Importa pandas para manipulación de datos
from sklearn.preprocessing import StandardScaler  # Importa StandardScaler para estandarización de variables numéricas
from modules.preprocessing import detectar_outliers_zscore, detectar_outliers_iqr, detectar_outliers_isolation_forest, encoding_categorias, mostrar_resumen_outliers  # Importa funciones personalizadas para limpieza

# =========================
# Preprocesamiento de los datos
# =========================

if st.session_state.get('pasar_a_limpieza') == True:  # Verifica si se puede avanzar a la etapa de limpieza
    st.title("Preprocesamiento de Datos")  # Muestra el título principal de la página

    archivos = list(st.session_state['archivos'].items())  # Obtiene la lista de archivos y sus datos del estado de sesión
    nombres = [nombre for nombre, _ in archivos]  # Extrae los nombres de los archivos
    tabs = st.tabs(nombres)  # Crea una pestaña para cada archivo

    for i, (nombre, datos) in enumerate(archivos):  # Itera sobre cada archivo y sus datos
        with tabs[i]:  # Selecciona la pestaña correspondiente
            df_original = datos['dataframe'].copy()  # Copia el DataFrame original para trabajar sobre él
            clasificacion = datos.get('clasificacion', {})  # Obtiene la clasificación de variables
            df_original.columns = df_original.columns.str.strip()  # Elimina espacios en los nombres de columnas
            if 'operaciones_realizadas' not in datos:  # Si no existe el registro de operaciones
                datos['operaciones_realizadas'] = []  # Inicializa la lista de operaciones realizadas

            etapa_tabs = st.tabs([  # Crea pestañas para cada etapa del preprocesamiento
                "Imputación de Nulos",
                "Borrado de Alta Cardinalidad",
                "Detección de Outliers",
                "Codificación y Estandarización"
            ])

            # =========================
            # Imputación de Nulos
            # =========================
            with etapa_tabs[0]:  # Pestaña de imputación de nulos
                if 'df_imputado' not in datos:  # Si no existe el DataFrame imputado
                    df_imputado = df_original.copy()  # Copia el DataFrame original
                    for columna, tipo in clasificacion.items():  # Itera sobre cada columna y su tipo
                        col = columna.strip()  # Elimina espacios en el nombre de la columna
                        if col in df_imputado.columns and df_imputado[col].isnull().sum() > 0:  # Si la columna tiene nulos
                            if tipo == "Numérica":  # Si la columna es numérica
                                df_imputado[col] = pd.to_numeric(df_imputado[col], errors='coerce')  # Convierte a numérico
                                mean_value = df_imputado[col].mean()  # Calcula el promedio
                                df_imputado[col] = df_imputado[col].fillna(mean_value)  # Imputa con el promedio
                                datos['operaciones_realizadas'].append({  # Registra la operación realizada
                                    'Archivo': nombre, 'Columna': col, 'Tipo': tipo,
                                    'Operación': f"Reemplazado NaN con promedio ({mean_value:.2f})"
                                })
                            elif tipo in ["Categórica", "Booleana", "Temporal"]:  # Si la columna es categórica, booleana o temporal
                                mode_series = df_imputado[col].mode()  # Obtiene el valor más común
                                mode_value = mode_series[0] if not mode_series.empty else None  # Toma el primer valor si existe
                                if mode_value is not None:  # Si hay valor más común
                                    df_imputado[col] = df_imputado[col].fillna(mode_value)  # Imputa con el valor más común
                                    datos['operaciones_realizadas'].append({  # Registra la operación realizada
                                        'Archivo': nombre, 'Columna': col, 'Tipo': tipo,
                                        'Operación': f"Reemplazado NaN con valor más común ({mode_value})"
                                    })
                                else:  # Si no se puede determinar el valor más común
                                    datos['operaciones_realizadas'].append({
                                        'Archivo': nombre, 'Columna': col, 'Tipo': tipo,
                                        'Operación': "No se pudo determinar el valor más común"
                                    })
                    datos['df_imputado'] = df_imputado  # Guarda el DataFrame imputado
                else:
                    df_imputado = datos['df_imputado']  # Recupera el DataFrame imputado

                st.markdown("#### Imputación de Nulos")  # Muestra subtítulo
                if datos['operaciones_realizadas']:  # Si se realizaron operaciones
                    st.dataframe(pd.DataFrame(datos['operaciones_realizadas']), use_container_width=True)  # Muestra las operaciones realizadas
                else:
                    st.info(f"No fue necesario realizar operaciones durante la limpieza para el archivo '{nombre}'.")  # Informa si no hubo operaciones

            # Identificar columnas categóricas y numéricas
            cat_attribs = [col.strip() for col, tipo in clasificacion.items() if tipo == "Categórica" and col.strip() in df_imputado.columns]  # Lista de columnas categóricas
            num_attribs = [col.strip() for col, tipo in clasificacion.items() if tipo == "Numérica" and col.strip() in df_imputado.columns]  # Lista de columnas numéricas

            # =========================
            # Borrado de Alta Cardinalidad
            # =========================
            with etapa_tabs[1]:  # Pestaña de borrado de alta cardinalidad
                df_card = datos.get('df_card', df_imputado.copy())  # Usa el DataFrame imputado o el de alta cardinalidad previo
                st.markdown("#### Borrado de Columnas de Alta Cardinalidad")  # Muestra subtítulo
                umbral = st.number_input(  # Permite al usuario definir el umbral de valores únicos
                    "Define el umbral de valores únicos para considerar alta cardinalidad:",
                    min_value=2, max_value=10000, value=100, step=1, key=f"umbral_cardinalidad_{nombre}"
                )
                columnas_alta_cardinalidad = [(col, df_card[col].nunique()) for col in df_card.columns if df_card[col].nunique() > umbral]  # Identifica columnas con alta cardinalidad
                if columnas_alta_cardinalidad:  # Si existen columnas con alta cardinalidad
                    st.warning(f"Las siguientes columnas tienen alta cardinalidad (> {umbral} valores únicos):")  # Advierte al usuario
                    df_cardinalidad = pd.DataFrame(columnas_alta_cardinalidad, columns=["Columna", "Valores Únicos"])  # DataFrame con columnas y valores únicos
                    st.dataframe(df_cardinalidad, use_container_width=True)  # Muestra el DataFrame
                    nombres_columnas = [col for col, _ in columnas_alta_cardinalidad]  # Lista de nombres de columnas a eliminar
                    columnas_a_eliminar = st.multiselect(  # Permite seleccionar columnas a eliminar
                        "Selecciona las columnas que deseas eliminar por alta cardinalidad:",
                        nombres_columnas,
                        placeholder="Selecciona una o varias columnas",
                        key=f"accion_cardinalidad_{nombre}"
                    )
                    if st.button("Eliminar columnas seleccionadas", key=f"eliminar_cardinalidad_{nombre}"):  # Botón para eliminar columnas
                        df_card = df_card.drop(columns=columnas_a_eliminar)  # Elimina las columnas seleccionadas
                        datos['df_card'] = df_card  # Guarda el DataFrame actualizado
                        st.success(f"Columnas eliminadas: {columnas_a_eliminar}")  # Muestra mensaje de éxito
                    else:
                        datos['df_card'] = df_card  # Guarda el DataFrame sin cambios
                else:
                    st.info("No se detectaron columnas con alta cardinalidad.")  # Informa si no hay columnas con alta cardinalidad
                    datos['df_card'] = df_card  # Guarda el DataFrame

            # Actualizar atributos categóricos y numéricos tras borrado de alta cardinalidad
            cat_attribs_card = [col for col in cat_attribs if col in datos['df_card'].columns]  # Columnas categóricas restantes
            num_attribs_card = [col for col in num_attribs if col in datos['df_card'].columns]  # Columnas numéricas restantes

            # =========================
            # Detección de Outliers
            # =========================
            with etapa_tabs[2]:  # Pestaña de detección de outliers
                df_outliers = datos['df_card'].copy()  # Copia el DataFrame tras alta cardinalidad
                st.markdown("#### Detección de Outliers")  # Muestra subtítulo
                metodo = st.selectbox("Método de detección", ["Z-Score", "IQR", "Isolation Forest"], key=f"metodo_{nombre}")  # Selección de método
                if metodo == "Z-Score":  # Si el método es Z-Score
                    outliers = detectar_outliers_zscore(df_outliers, num_attribs_card)  # Detecta outliers con Z-Score
                elif metodo == "IQR":  # Si el método es IQR
                    outliers = detectar_outliers_iqr(df_outliers, num_attribs_card)  # Detecta outliers con IQR
                else:  # Si el método es Isolation Forest
                    outliers = detectar_outliers_isolation_forest(df_outliers, num_attribs_card)  # Detecta outliers con Isolation Forest
                mostrar_resumen_outliers(df_outliers, num_attribs_card, outliers, metodo)  # Muestra resumen de outliers
                accion = st.radio("¿Qué hacer con los outliers?", [  # Permite elegir acción sobre outliers
                    "Conservar todos los datos",
                    "Excluir outliers del análisis"
                ], key=f"accion_outliers_{nombre}")
                if accion == "Excluir outliers del análisis":  # Si se elige excluir outliers
                    df_limpio = df_outliers[~outliers].copy()  # Elimina los outliers
                    st.success(f"Eliminados {outliers.sum()} outliers")  # Muestra mensaje de éxito
                else:
                    df_limpio = df_outliers.copy()  # Conserva todos los datos
                datos['df_limpio_outliers'] = df_limpio  # Guarda el DataFrame limpio

            # Actualizar atributos categóricos y numéricos tras outliers
            cat_attribs_limpio = [col for col in cat_attribs_card if col in datos['df_limpio_outliers'].columns]  # Columnas categóricas finales
            num_attribs_limpio = [col for col in num_attribs_card if col in datos['df_limpio_outliers'].columns]  # Columnas numéricas finales

            # =========================
            # Codificación y Estandarización
            # =========================
            with etapa_tabs[3]:  # Pestaña de codificación y estandarización
                st.markdown("#### Codificación y Estandarización de Datos")  # Muestra subtítulo
                df_limpio = datos['df_limpio_outliers']  # DataFrame limpio tras outliers
                if cat_attribs_limpio:  # Si existen columnas categóricas
                    df_procesado, columnas_finales = encoding_categorias(df_limpio, cat_attribs_limpio, num_attribs_limpio, nombre)  # Codifica variables categóricas
                    st.success(f"Codificadas {len(cat_attribs_limpio)} variable(s) categórica(s)")  # Muestra mensaje de éxito
                else:  # Si no hay columnas categóricas
                    scaler = StandardScaler()  # Inicializa el escalador
                    df_procesado = df_limpio.copy()  # Copia el DataFrame limpio
                    df_procesado[num_attribs_limpio] = scaler.fit_transform(df_limpio[num_attribs_limpio])  # Estandariza variables numéricas
                    columnas_finales = num_attribs_limpio  # Solo columnas numéricas
                st.dataframe(df_procesado.head(), use_container_width=True)  # Muestra las primeras filas del DataFrame procesado
                datos['dataframe_final'] = df_procesado  # Guarda el DataFrame final procesado

            # =========================
            # Guardado de resultados intermedios
            # =========================
            datos['dataframe_limpio'] = datos['df_limpio_outliers']  # Guarda el DataFrame limpio tras imputación y outliers

    st.session_state['pasar_a_analisis'] = True  # Permite avanzar a la siguiente etapa (análisis)
else:
    st.warning("Primero completa la sección de Análisis Exploratorio de Datos.")  # Advierte si no se ha completado la etapa anterior
    st.session_state['pasar_a_analisis'] = False  # No permite avanzar a la etapa de análisis