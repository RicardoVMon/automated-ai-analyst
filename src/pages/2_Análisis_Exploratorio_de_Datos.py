import streamlit as st, pandas as pd
from modules.preprocessing import describir_archivo, clasificar_variables, generar_perfilado
import time

model = st.session_state.get('model', None)

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

            st.markdown("#### Perfilado Inicial de Datos")
            with st.spinner("Generando reporte de perfilado..."):
                if 'perfilado' not in st.session_state['archivos'][nombre]:
                    
                        profile = generar_perfilado(df, clasificacion)
                        st.session_state['archivos'][nombre]['perfilado'] = profile
                else:
                    profile = st.session_state['archivos'][nombre]['perfilado']

                if profile:
                
                    st.components.v1.html(profile.to_html(), height=600, scrolling=True)
                    # correlaciones = profile.get_description().correlations # así se accede a las correlaciones
                else:
                    st.warning(f"No se pudo generar el perfilado para {nombre}")
        else:
            st.warning(f"No se pudieron clasificar las variables de {nombre}")

        if nombre == list(st.session_state['archivos'].keys())[-1]:
            st.session_state['pasar_a_limpieza'] = True
else:
    st.warning("Primero completa la sección de Carga de Archivos.")