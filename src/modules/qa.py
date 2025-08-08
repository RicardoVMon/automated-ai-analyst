import streamlit as st
import time

def inicializar_chat():
    """Inicializa el chat con el contexto de los insights"""
    if 'chat_iniciado' not in st.session_state:
        st.session_state.chat_iniciado = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'gemini_chat' not in st.session_state:
        st.session_state.gemini_chat = None

def preparar_contexto_inicial():
    """Prepara el contexto inicial con los insights"""
    insights_texto = st.session_state.get('insights_finales', '')
    
    contexto_inicial = f"""
    Eres un analista de datos experto que ha realizado un análisis completo de clustering y detección de outliers. 
    A continuación tienes los resultados del análisis que acabas de completar:

    CONTEXTO DEL ANÁLISIS:
    {insights_texto}

    Tu rol es:
    1. Responder preguntas sobre estos resultados de manera clara y profesional
    2. Explicar los patrones encontrados en los datos
    3. Sugerir acciones basadas en los clusters identificados
    4. Ayudar a interpretar las correlaciones y outliers
    5. Proporcionar recomendaciones de negocio cuando sea apropiado

    Responde de manera conversacional, como si fueras un consultor de datos experimentado.
    Si te preguntan algo que no está en los datos del análisis, indícalo claramente.
    """
    
    return contexto_inicial

def enviar_mensaje_inicial():
    """Envía el contexto inicial al modelo y obtiene el saludo"""
    try:
        if st.session_state.gemini_chat is None:
            model = st.session_state.get('model')
            if model is None:
                return False, "No se pudo cargar el modelo Gemini."
            
            st.session_state.gemini_chat = model.start_chat(history=[])
        
        # Enviar contexto inicial
        contexto = preparar_contexto_inicial()
        
        mensaje_inicial = f"""
        {contexto}
        
        Ahora saluda al usuario y pregúntale qué aspecto del análisis le gustaría explorar. 
        Menciona brevemente que tienes información sobre clusters, correlaciones y outliers disponible.
        """
        
        response = st.session_state.gemini_chat.send_message(mensaje_inicial)
        
        # Agregar saludo del asistente al historial
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response.text,
            "timestamp": time.time()
        })
        
        return True, response.text
        
    except Exception as e:
        return False, f"Error al inicializar el chat: {e}"

def mostrar_mensaje(mensaje, role):
    """Muestra un mensaje en el chat con el formato apropiado"""
    if role == "user":
        with st.chat_message("user"):
            st.write(mensaje["content"])
    else:
        with st.chat_message("assistant"):
            st.write(mensaje["content"])

def procesar_mensaje_usuario(user_input):
    """Procesa el mensaje del usuario y obtiene la respuesta"""
    try:
        # Enviar mensaje al modelo
        response = st.session_state.gemini_chat.send_message(user_input)
        
        # Agregar intercambio al historial
        st.session_state.chat_history.extend([
            {
                "role": "user", 
                "content": user_input,
                "timestamp": time.time()
            },
            {
                "role": "assistant", 
                "content": response.text,
                "timestamp": time.time()
            }
        ])
        
        return True, response.text
    
    except Exception as e:
        return False, f"Error al enviar mensaje: {e}"