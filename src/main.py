# ...existing code...

import streamlit as st

st.set_page_config(page_title="AutoData", layout="centered", initial_sidebar_state="expanded")

# --- Configuración inicial ---
st.title("AutoData")
st.subheader("Sistema de Análisis Automático de Datos")

# --- Configuración de API Key ---
st.markdown("### Configuración inicial")

# Verificar si ya existe la API key en session state
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

# Input para la API key
api_key = st.text_input(
    "Ingresa tu API Key de Google Gemini:",
    value=st.session_state.gemini_api_key,
    type="password",
    help="Necesitas una API Key de Google Gemini para usar las funciones de IA del sistema. Obtén tu clave en: https://aistudio.google.com/app/apikey"
)

# Guardar en session state cuando cambie
if api_key != st.session_state.gemini_api_key:
    st.session_state.gemini_api_key = api_key

# Mostrar estado de la configuración
if st.session_state.gemini_api_key:
    st.success("API Key configurada correctamente")
else:
    st.warning("Necesitas configurar tu API Key de Google Gemini para usar todas las funciones del sistema")

st.divider()

st.markdown("""
### Haciendo el Machine Learning fácil para todos

**AutoData** democratiza el análisis de datos avanzado y el Machine Learning, cerrando la brecha técnica 
entre conceptos complejos y usuarios sin experiencia especializada. Nuestro sistema utiliza **Inteligencia 
Artificial generativa** para automatizar completamente el proceso de análisis, desde la carga de datos 
hasta la generación de insights en lenguaje natural.

#### ¿Qué hace AutoData por ti?

- **Análisis Automático**: Clasificación inteligente de variables y perfilado completo de tus datos
- **Detección de Outliers**: Múltiples algoritmos avanzados (Z-Score, IQR, Isolation Forest) aplicados automáticamente
- **Clustering Inteligente**: K-means y DBSCAN con optimización automática de parámetros
- **Insights con IA**: Generación automática de conclusiones y recomendaciones en lenguaje comprensible
- **Chat Inteligente**: Consulta natural sobre tus resultados, como si hablaras con un analista experto

#### Flujo de trabajo simplificado:

1. **Carga de Archivos** → Sube tus CSV/Excel
2. **Análisis Exploratorio** → Revisa el perfilado automático  
3. **Limpieza de Datos** → Detecta y trata outliers automáticamente
4. **Machine Learning** → Aplica clustering no supervisado
5. **Consultas y Chat** → Explora tus resultados conversacionalmente

**Tecnologías**: Streamlit | scikit-learn | Google Gemini | Plotly | pandas
""")

# Información de navegación
if st.session_state.gemini_api_key:
    st.info("""
    **Usa la barra lateral** para navegar entre las diferentes etapas del análisis.

    **Flujo recomendado**: Sigue el orden numérico de las páginas para obtener los mejores resultados.
    """)
else:
    st.info("""
    **Configura tu API Key** de Google Gemini arriba para comenzar a usar el sistema.
    
    **Una vez configurada**, usa la barra lateral para navegar entre las diferentes etapas del análisis.
    """)