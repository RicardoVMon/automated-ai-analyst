# =========================
# Importación de librerías y módulos
# =========================

import streamlit as st  # Importa la librería Streamlit para crear la interfaz web
from modules.load import filtrar_duplicados, leer_archivos  # Importa las funciones personalizadas para filtrar duplicados y leer archivos

# =========================
# Título de la página
# =========================

st.title("Carga de Archivos")  # Muestra el título principal de la página en la interfaz

# =========================
# --- Carga de archivos ---
# =========================

uploaded_files = st.file_uploader(  # Crea un componente para cargar archivos en la interfaz
    "Carga tu archivo CSV o Excel",  # Texto que se muestra al usuario
    type=["csv", "xlsx"],  # Limita los tipos de archivos permitidos a CSV y Excel
    accept_multiple_files=True  # Permite la carga de múltiples archivos a la vez
)

# =========================
# Selección y procesamiento de archivos cargados
# =========================

if uploaded_files:  # Si el usuario ha cargado al menos un archivo
    uploaded_files = filtrar_duplicados(uploaded_files)  # Elimina archivos duplicados usando la función personalizada
    archivos = leer_archivos(uploaded_files, st.session_state.get('archivos', {}))  # Lee los archivos y retorna un diccionario con el nombre y su dataframe

    if len(archivos) > 0:  # Si se logró cargar al menos un archivo correctamente
        st.success(f"{len(archivos)} archivo(s) cargado(s) correctamente")  # Muestra un mensaje de éxito con la cantidad de archivos cargados

        # Si hay más de un archivo, permitir selección
        if len(archivos) > 1:  # Si hay más de un archivo cargado
            st.subheader("Selecciona los archivos para analizar")  # Muestra un subtítulo para la selección de archivos

            st.markdown(  # Muestra instrucciones y recomendaciones para la selección de archivos
                """
                A continuación, seleccione los archivos que desea incluir en el análisis conjunto.
                **Nota:** Los archivos que no sean seleccionados serán excluidos del análisis.
                """
            )

            seleccionados = st.multiselect(  # Permite al usuario seleccionar uno o varios archivos de la lista
                "Elige los archivos relacionados:",  # Texto de la pregunta
                options=list(archivos.keys()),  # Opciones disponibles: nombres de los archivos cargados
                default=list(archivos.keys()),  # Por defecto, todos los archivos están seleccionados
                placeholder="Selecciona uno o varios archivos..."  # Texto de ayuda cuando no hay selección
            )

            if seleccionados:  # Si el usuario seleccionó al menos un archivo
                # Solo guarda los seleccionados
                st.session_state['archivos'] = {nombre: archivos[nombre] for nombre in seleccionados}  # Guarda solo los archivos seleccionados en el estado de la sesión
                st.session_state['pasar_a_perfilado'] = True  # Permite avanzar a la siguiente etapa (perfilado)
                st.success(f"Archivos seleccionados para el análisis: {', '.join(seleccionados)}")  # Muestra los archivos seleccionados
            else:  # Si no se seleccionó ningún archivo
                st.session_state['archivos'] = {}  # Limpia la variable de archivos en el estado de la sesión
                st.session_state['pasar_a_perfilado'] = False  # No permite avanzar a la siguiente etapa
                st.warning("No se ha seleccionado ningún archivo relacionado. No se puede continuar con el análisis.")  # Advierte al usuario que debe seleccionar al menos uno
        else:  # Si solo hay un archivo cargado
            # Si solo hay uno, se guarda directamente
            st.session_state['archivos'] = archivos  # Guarda el único archivo cargado en el estado de la sesión
            st.session_state['pasar_a_perfilado'] = True  # Permite avanzar a la siguiente etapa

    else:  # Si no se pudo cargar ningún archivo válido
        st.warning("No se pudo cargar ningún archivo válido.")  # Advierte al usuario que no se cargó ningún archivo válido
else:  # Si no se han cargado archivos
    st.session_state['pasar_a_perfilado'] = False  # No permite avanzar a la siguiente etapa
    st.session_state['pasar_a_limpieza'] = False  # No permite avanzar a la etapa de limpieza
    st.session_state['pasar_a_analisis'] = False  # No permite avanzar a la etapa de análisis
    st.info("Esperando archivos...")  # Informa al usuario que debe cargar archivos