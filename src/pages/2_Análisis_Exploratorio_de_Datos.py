import streamlit as st, pandas as pd
from modules.settings import cargar_api_gemini
from modules.preprocessing import describir_archivo, clasificar_variables, generar_perfilado

model = cargar_api_gemini()

# Perfilado de los datos
if st.session_state.get('pasar_a_perfilado') == True:
    st.title("Análisis Exploratorio de Datos")

    archivos = list(st.session_state['archivos'].items())
    nombres = [nombre for nombre, _ in archivos]
    tabs = st.tabs(nombres)

    for i, (nombre, datos) in enumerate(archivos):
        with tabs[i]:
            df = datos['dataframe']
            st.markdown(f"## {nombre}")

            if datos.get('descripcion') is None:
                with st.spinner("Describiendo archivo..."):
                    try:
                        descripcion = describir_archivo(df, model)
                        st.session_state['archivos'][nombre]['descripcion'] = descripcion
                    except Exception as e:
                        st.error(f"Error al describir el archivo de {nombre}: {e}")
                        st.session_state['archivos'][nombre]['descripcion'] = {}

            descripcion = st.session_state['archivos'][nombre].get('descripcion', {})

            if datos.get('clasificacion') is None:
                with st.spinner("Clasificando variables..."):
                    try:
                        clasificacion = clasificar_variables(df, model)
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

                st.markdown("#### Perfilado Inicial de Datos")
                with st.spinner("Generando reporte de perfilado..."):
                    if 'perfilado' not in st.session_state['archivos'][nombre]:
                        profile = generar_perfilado(df, clasificacion)
                        st.session_state['archivos'][nombre]['perfilado'] = profile
                    else:
                        profile = st.session_state['archivos'][nombre]['perfilado']

                    if profile:
                        st.components.v1.html(profile.to_html(), height=600, scrolling=True)
                        html_bytes = profile.to_html().encode('utf-8')
                        st.download_button(
                            label="Descargar perfilado en HTML",
                            data=html_bytes,
                            file_name=f"{nombre}_perfilado.html",
                            mime="text/html"
                        )
                    else:
                        st.warning(f"No se pudo generar el perfilado para {nombre}")
            else:
                st.warning(f"No se pudieron clasificar las variables de {nombre}")

            if nombre == list(st.session_state['archivos'].keys())[-1]:
                st.session_state['pasar_a_limpieza'] = True
else:
    st.warning("Primero completa la sección de Carga de Archivos.")