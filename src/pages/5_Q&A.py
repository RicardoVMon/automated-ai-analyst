# =========================
# Importación de librerías y módulos
# =========================

import streamlit as st  # Importa Streamlit para la interfaz web
from modules.qa import inicializar_chat, enviar_mensaje_inicial, procesar_mensaje_usuario, mostrar_mensaje  # Importa funciones personalizadas para el chat

# =========================
# Punto de entrada y control de acceso
# =========================

if st.session_state.get('pasar_a_q&a') == True:  # Verifica si se puede acceder a la etapa de Q&A
    st.title("Asistente de Análisis de Datos")  # Muestra el título principal de la página
    st.markdown("Pregunta sobre los resultados de tu análisis de clustering y outliers")  # Explicación breve
    
    # =========================
    # Reinicio del chat si es necesario
    # =========================
    if st.session_state.get('reiniciar_chat', False):  # Si el flag de reinicio está activo
        for key in ['chat_iniciado', 'chat_history']:  # Itera sobre las claves del chat
            if key in st.session_state:  # Si la clave existe en el estado de sesión
                del st.session_state[key]  # Elimina la clave
        st.session_state['reiniciar_chat'] = False  # Desactiva el flag de reinicio
        st.rerun()  # Recarga la página para reiniciar el chat

    # =========================
    # Inicialización del chat
    # =========================
    inicializar_chat()  # Inicializa el chat y variables necesarias
    
    # =========================
    # Envío del mensaje inicial si el chat no está iniciado
    # =========================
    if not st.session_state.chat_iniciado:  # Si el chat no ha sido iniciado
        with st.spinner("Inicializando asistente..."):  # Muestra spinner de carga
            success, message = enviar_mensaje_inicial()  # Envía el mensaje inicial al asistente
            
            if success:  # Si el mensaje inicial fue exitoso
                st.session_state.chat_iniciado = True  # Marca el chat como iniciado
                st.rerun()  # Recarga para mostrar el mensaje inicial
            else:  # Si hubo error al iniciar el chat
                st.error(f"{message}")  # Muestra el error
                st.stop()  # Detiene la ejecución

    # =========================
    # Visualización del historial de chat
    # =========================
    for mensaje in st.session_state.chat_history:  # Itera sobre el historial de mensajes
        mostrar_mensaje(mensaje, mensaje["role"])  # Muestra cada mensaje según su rol (usuario o asistente)
    
    # =========================
    # Entrada y procesamiento de preguntas del usuario
    # =========================
    user_input = st.chat_input("Escribe tu pregunta sobre el análisis...")  # Campo de entrada para el usuario
    
    if user_input:  # Si el usuario ingresa una pregunta
        # Mostrar mensaje del usuario inmediatamente
        with st.chat_message("user"):  # Muestra el mensaje como usuario
            st.write(user_input)
        
        # Procesar y mostrar respuesta
        with st.chat_message("assistant"):  # Muestra la respuesta como asistente
            with st.spinner("Pensando..."):  # Muestra spinner de carga
                success, response = procesar_mensaje_usuario(user_input)  # Procesa la pregunta y obtiene la respuesta
                
                if success:  # Si la respuesta fue exitosa
                    st.write(response)  # Muestra la respuesta
                else:  # Si hubo error
                    st.error(f"{response}")  # Muestra el error
else:
    st.warning("Primero completa el análisis de clustering en la sección anterior.")  # Advierte si no se ha completado