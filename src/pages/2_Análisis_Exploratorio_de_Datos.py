# =========================
# Importación de librerías y módulos
# =========================

import streamlit as st, pandas as pd  # Importa Streamlit para la interfaz y pandas para manipulación de datos
from modules.settings import cargar_api_gemini  # Importa función para cargar el modelo de API Gemini
from modules.preprocessing import describir_archivo, clasificar_variables, generar_perfilado  # Importa funciones de procesamiento

# =========================
# Carga del modelo de IA
# =========================

model = cargar_api_gemini()  # Inicializa el modelo de IA Gemini para análisis y perfilado

# =========================
# Perfilado de los datos
# =========================

if st.session_state.get('pasar_a_perfilado') == True:  # Verifica si se puede avanzar al perfilado de datos
    st.title("Análisis Exploratorio de Datos")  # Muestra el título principal de la página

    archivos = list(st.session_state['archivos'].items())  # Obtiene la lista de archivos y sus datos del estado de sesión
    nombres = [nombre for nombre, _ in archivos]  # Extrae los nombres de los archivos
    tabs = st.tabs(nombres)  # Crea una pestaña para cada archivo

    for i, (nombre, datos) in enumerate(archivos):  # Itera sobre cada archivo y sus datos
        with tabs[i]:  # Selecciona la pestaña correspondiente
            df = datos['dataframe']  # Obtiene el DataFrame del archivo
            st.markdown(f"## {nombre}")  # Muestra el nombre del archivo como subtítulo

            # =========================
            # Descripción del archivo
            # =========================
            if datos.get('descripcion') is None:  # Si no existe descripción previa
                with st.spinner("Describiendo archivo..."):  # Muestra spinner de carga
                    try:
                        descripcion = describir_archivo(df, model)  # Genera la descripción usando IA
                        st.session_state['archivos'][nombre]['descripcion'] = descripcion  # Guarda la descripción en sesión
                    except Exception as e:
                        st.error(f"Error al describir el archivo de {nombre}: {e}")  # Muestra error si falla
                        st.session_state['archivos'][nombre]['descripcion'] = {}  # Guarda descripción vacía

            descripcion = st.session_state['archivos'][nombre].get('descripcion', {})  # Recupera la descripción

            # =========================
            # Clasificación de variables
            # =========================
            if datos.get('clasificacion') is None:  # Si no existe clasificación previa
                with st.spinner("Clasificando variables..."):  # Muestra spinner de carga
                    try:
                        clasificacion = clasificar_variables(df, model)  # Clasifica las variables usando IA
                        st.session_state['archivos'][nombre]['clasificacion'] = clasificacion  # Guarda la clasificación en sesión
                    except Exception as e:
                        st.error(f"Error al clasificar variables de {nombre}: {e}")  # Muestra error si falla
                        st.session_state['archivos'][nombre]['clasificacion'] = {}  # Guarda clasificación vacía

            clasificacion = st.session_state['archivos'][nombre].get('clasificacion', {})  # Recupera la clasificación

            # =========================
            # Visualización de resultados
            # =========================
            if descripcion:  # Si hay descripción disponible
                st.markdown("#### Descripción del archivo")  # Muestra subtítulo
                st.markdown(descripcion)  # Muestra la descripción generada

            if clasificacion:  # Si hay clasificación disponible
                st.dataframe(df.head(), use_container_width=True)  # Muestra las primeras filas del DataFrame

                st.markdown("#### Clasificación detectada")  # Muestra subtítulo
                st.table(
                    pd.DataFrame.from_dict(clasificacion, orient='index', columns=['Tipo'])  # Convierte la clasificación a DataFrame
                    .rename_axis('Columna')  # Renombra el índice como 'Columna'
                    .reset_index()  # Resetea el índice para mostrarlo como columna
                )

                st.markdown("#### Perfilado Inicial de Datos")  # Muestra subtítulo
                with st.spinner("Generando reporte de perfilado..."):  # Muestra spinner de carga
                    if 'perfilado' not in st.session_state['archivos'][nombre]:  # Si no existe perfilado previo
                        profile = generar_perfilado(df, clasificacion)  # Genera el perfilado usando la función correspondiente
                        st.session_state['archivos'][nombre]['perfilado'] = profile  # Guarda el perfilado en sesión
                    else:
                        profile = st.session_state['archivos'][nombre]['perfilado']  # Recupera el perfilado existente

                    if profile:  # Si el perfilado fue generado correctamente
                        st.components.v1.html(profile.to_html(), height=600, scrolling=True)  # Muestra el perfilado en HTML
                        html_bytes = profile.to_html().encode('utf-8')  # Convierte el perfilado a bytes para descarga
                        st.download_button(
                            label="Descargar perfilado en HTML",  # Etiqueta del botón de descarga
                            data=html_bytes,  # Datos a descargar
                            file_name=f"{nombre}_perfilado.html",  # Nombre del archivo de descarga
                            mime="text/html"  # Tipo MIME
                        )
                    else:
                        st.warning(f"No se pudo generar el perfilado para {nombre}")  # Advierte si no se pudo generar el perfilado
            else:
                st.warning(f"No se pudieron clasificar las variables de {nombre}")  # Advierte si no se pudo clasificar

            # =========================
            # Control de avance de etapa
            # =========================
            if nombre == list(st.session_state['archivos'].keys())[-1]:  # Si es el último archivo de la lista
                st.session_state['pasar_a_limpieza'] = True  # Permite avanzar a la siguiente etapa (limpieza)
else:
    st.warning("Primero completa la sección de Carga de Archivos.")  # Advierte si no se ha completado la