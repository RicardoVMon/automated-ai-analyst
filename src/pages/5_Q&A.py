import streamlit as st
from modules.qa import inicializar_chat, enviar_mensaje_inicial, procesar_mensaje_usuario, mostrar_mensaje


# --- PUNTO DE ENTRADA ---
if st.session_state.get('pasar_a_q&a') == True:
    st.title("Asistente de Análisis de Datos")
    st.markdown("Pregunta sobre los resultados de tu análisis de clustering y outliers")
    
    # Reiniciar chat si el flag está activo
    if st.session_state.get('reiniciar_chat', False):
        for key in ['chat_iniciado', 'chat_history']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state['reiniciar_chat'] = False
        st.rerun()

    # Inicializar chat
    inicializar_chat()
    
    # Iniciar chat si no se ha iniciado
    if not st.session_state.chat_iniciado:
        with st.spinner("Inicializando asistente..."):
            success, message = enviar_mensaje_inicial()
            
            if success:
                st.session_state.chat_iniciado = True
                st.rerun()  # Recargar para mostrar el mensaje inicial
            else:
                st.error(f"{message}")
                st.stop()
    
    # Mostrar historial de chat
    for mensaje in st.session_state.chat_history:
        mostrar_mensaje(mensaje, mensaje["role"])
    
    # Input del usuario
    user_input = st.chat_input("Escribe tu pregunta sobre el análisis...")
    
    if user_input:
        # Mostrar mensaje del usuario inmediatamente
        with st.chat_message("user"):
            st.write(user_input)
        
        # Procesar y mostrar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                success, response = procesar_mensaje_usuario(user_input)
                
                if success:
                    st.write(response)
                else:
                    st.error(f"{response}")
else:
    st.warning("Primero completa el análisis de clustering en la sección anterior.")