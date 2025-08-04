import streamlit as st
from modules.load import filtrar_duplicados, leer_archivos

st.title("Carga de Archivos")

# --- Carga de archivos ---
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
                """
                A continuación, seleccione los archivos que desea incluir en el análisis conjunto.

                **Recomendación:** Seleccione archivos que estén relacionados entre sí, es decir, que compartan claves comunes, tengan la misma estructura o puedan combinarse sin problemas.

                **Nota:** Los archivos que no sean seleccionados serán excluidos del análisis.
                """
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
    st.session_state['pasar_a_limpieza'] = False
    st.session_state['pasar_a_analisis'] = False
    st.info("Esperando archivos...")