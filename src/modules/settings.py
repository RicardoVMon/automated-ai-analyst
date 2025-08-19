import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

def cargar_api_gemini():
    try:
        if 'model' not in st.session_state:
            load_dotenv()
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            st.session_state['model'] = genai.GenerativeModel("gemini-1.5-pro")
            return st.session_state['model']
        else:
            return st.session_state['model']
    except Exception as e:
        return None