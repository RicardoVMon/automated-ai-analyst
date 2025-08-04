import streamlit as st
import pandas as pd

# Limpieza de los datos
if st.session_state.get('pasar_a_limpieza') == True:
    st.title("Limpieza de Datos")

    for nombre, datos in st.session_state['archivos'].items():
        df = datos['dataframe']
        clasificacion = datos.get('clasificacion', {})

        if not clasificacion:
            st.warning(f"No se puede limpiar el archivo '{nombre}' porque no tiene clasificación.")
            continue

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
        st.session_state['archivos'][nombre]['dataframe_limpio'] = df

    # Mostrar resumen de operaciones realizadas por archivo
    st.markdown("### Resumen de Operaciones Realizadas")
    for nombre, datos in st.session_state['archivos'].items():
        st.markdown(f"#### Archivo: {nombre}")
        operaciones = datos.get('operaciones_realizadas', [])
        if operaciones:
            operaciones_df = pd.DataFrame(operaciones)
            st.dataframe(operaciones_df, use_container_width=True)
        else:
            st.info(f"No se realizaron operaciones durante la limpieza para el archivo '{nombre}'.")

    st.session_state['pasar_a_analisis'] = True
else:
    st.warning("Primero completa la sección de Análisis Exploratorio de Datos.")
    st.session_state['pasar_a_analisis'] = False