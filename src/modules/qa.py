import streamlit as st
import time
from modules.settings import cargar_api_gemini

model = cargar_api_gemini()  # Inicializar modelo de IA Gemini

def inicializar_chat():
    """
    Inicializa las variables de estado necesarias para el chat.
    
    Parámetros:
    -----------
    Ninguno
    
    Retorna:
    --------
    Ninguno (modifica st.session_state)
    
    Proceso:
    --------
    1. Verifica si el chat ya está iniciado
    2. Inicializa el historial de conversación vacío
    3. Prepara el objeto de chat de Gemini
    """
    # Verificar si ya se inició el chat anteriormente
    if 'chat_iniciado' not in st.session_state:
        st.session_state.chat_iniciado = False
    
    # Inicializar historial de mensajes vacío
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Inicializar objeto de chat de Gemini
    if 'gemini_chat' not in st.session_state:
        st.session_state.gemini_chat = None

def preparar_contexto_inicial():
    """
    Prepara el contexto inicial con todos los insights generados previamente.
    
    Parámetros:
    -----------
    Ninguno
    
    Retorna:
    --------
    str
        Contexto inicial completo con instrucciones y datos del análisis
    
    Proceso:
    --------
    1. Recopila todos los insights finales del análisis
    2. Formatea el texto con información del archivo y variables
    3. Crea prompt de contexto con rol e instrucciones
    """
    # Recopilar todos los insights finales generados
    insights_texto = "\n\n".join(
    f"Archivo: {i['archivo']} | Variables: {', '.join(i['variables'])}\n{i['insight']}"
    for i in st.session_state.get('insights_finales', [])
)
    
    # Crear contexto inicial con rol e instrucciones para la IA
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
    """
    Envía el contexto inicial al modelo y obtiene el mensaje de saludo.
    
    Parámetros:
    -----------
    Ninguno
    
    Retorna:
    --------
    tuple
        - bool: True si fue exitoso, False si hubo error
        - str: Mensaje de saludo del asistente o mensaje de error
    
    Proceso:
    --------
    1. Inicia una nueva sesión de chat con Gemini
    2. Envía el contexto inicial con instrucciones
    3. Obtiene y almacena el saludo del asistente
    4. Maneja errores de conexión o API
    """
    try:
        # Iniciar nueva sesión de chat con Gemini (historial vacío)
        st.session_state.gemini_chat = model.start_chat(history=[])
        
        # Obtener contexto inicial preparado
        contexto = preparar_contexto_inicial()
        
        # Crear mensaje inicial con contexto e instrucción de saludo
        mensaje_inicial = f"""
        {contexto}
        
        Ahora saluda al usuario y pregúntale qué aspecto del análisis le gustaría explorar. 
        """
        
        # Enviar mensaje inicial al modelo y obtener respuesta
        response = st.session_state.gemini_chat.send_message(mensaje_inicial)
        
        # Agregar saludo del asistente al historial de chat
        st.session_state.chat_history.append({
            "role": "assistant",           # Rol del mensaje (asistente)
            "content": response.text,      # Contenido de la respuesta
            "timestamp": time.time()       # Marca de tiempo del mensaje
        })
        
        return True, response.text         # Retorno exitoso con saludo
        
    except Exception as e:
        # Capturar y retornar cualquier error de inicialización
        return False, f"Error al inicializar el chat: {e}"

def mostrar_mensaje(mensaje, role):
    """
    Muestra un mensaje en la interfaz de chat con el formato apropiado.
    
    Parámetros:
    -----------
    mensaje : dict
        Diccionario con 'content' del mensaje a mostrar
    role : str
        Rol del mensaje ('user' o 'assistant')
    
    Retorna:
    --------
    Ninguno (muestra en interfaz)
    
    Proceso:
    --------
    1. Identifica si es mensaje de usuario o asistente
    2. Usa el componente apropiado de Streamlit
    3. Renderiza el contenido del mensaje
    """
    # Mostrar mensaje de usuario con icono correspondiente
    if role == "user":
        with st.chat_message("user"):
            st.write(mensaje["content"])    # Mostrar contenido del usuario
    else:
        # Mostrar mensaje de asistente con icono correspondiente
        with st.chat_message("assistant"):
            st.write(mensaje["content"])    # Mostrar contenido del asistente

def procesar_mensaje_usuario(user_input):
    """
    Procesa el mensaje del usuario y obtiene la respuesta del asistente.
    
    Parámetros:
    -----------
    user_input : str
        Mensaje escrito por el usuario
    
    Retorna:
    --------
    tuple
        - bool: True si fue exitoso, False si hubo error
        - str: Respuesta del asistente o mensaje de error
    
    Proceso:
    --------
    1. Envía el mensaje del usuario al modelo
    2. Obtiene la respuesta del asistente
    3. Almacena ambos mensajes en el historial
    4. Maneja errores de comunicación
    """
    try:
        # Enviar mensaje del usuario al modelo y obtener respuesta
        response = st.session_state.gemini_chat.send_message(user_input)
        
        # Agregar tanto el mensaje del usuario como la respuesta al historial
        st.session_state.chat_history.extend([
            {
                "role": "user",                # Mensaje del usuario
                "content": user_input,         # Contenido del mensaje del usuario
                "timestamp": time.time()       # Marca de tiempo
            },
            {
                "role": "assistant",           # Respuesta del asistente
                "content": response.text,      # Contenido de la respuesta
                "timestamp": time.time()       # Marca de tiempo
            }
        ])
        
        return True, response.text             # Retorno exitoso con respuesta
    
    except Exception as e:
        # Capturar y retornar cualquier error de comunicación
        return False, f"Error al enviar mensaje: {e}"