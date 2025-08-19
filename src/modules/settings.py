import streamlit as st  # Importa Streamlit para el manejo de estado de sesión y mensajes
import os  # Importa os para acceder a variables de entorno
from dotenv import load_dotenv  # Importa load_dotenv para cargar variables de entorno desde un archivo .env
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
        if 'model' not in st.session_state:  # Si el modelo no está en el estado de sesión
            genai.configure(api_key=st.session_state['gemini_api_key'])  # Configura la API key de Google
            st.session_state['model'] = genai.GenerativeModel("gemini-1.5-pro")  # Crea y guarda el modelo en sesión
            return st.session_state['model']  # Retorna el modelo
        else:
            return st.session_state['model']  # Si ya existe, retorna el modelo existente
    except Exception as e:  # Si ocurre algún error
        return None  # Retorna None en caso de error