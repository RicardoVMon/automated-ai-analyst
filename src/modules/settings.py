import streamlit as st  # Importa Streamlit para el manejo de estado de sesión y mensajes
import google.generativeai as genai  # Importa la librería de Google Generative AI

def cargar_api_gemini():
    """
    Inicializa y carga el modelo de IA Gemini de Google en el estado de sesión de Streamlit.

    Qué hace:
        - Carga la clave de API de Google desde las variables de entorno.
        - Configura la librería generativeai con la clave de API.
        - Crea una instancia del modelo 'gemini-1.5-pro' y la guarda en el estado de sesión de Streamlit.
        - Si el modelo ya está cargado en sesión, lo reutiliza.

    Parámetros:
        - No recibe parámetros de entrada.

    Retorna:
        - El objeto del modelo generativo Gemini si la inicialización es exitosa.
        - None si ocurre algún error durante la inicialización.
    """
    try:
        genai.configure(api_key=st.session_state['gemini_api_key'])  # Configura la API key de Google
        return genai.GenerativeModel("gemini-1.5-pro")  # Crea y guarda el modelo en sesión
    except Exception as e:  # Si ocurre algún error
        return None  # Retorna None en caso de error