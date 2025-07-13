import os
from dotenv import load_dotenv
import google.generativeai as genai

def cargar_api_gemini():
    try:
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        print(f"Error al cargar API: {e}")
        return None

def chat_consola():
    model = cargar_api_gemini()
    if model is None:
        print("No se pudo cargar el modelo.")
        return

    chat = model.start_chat(history=[])  # Nueva conversación

    print("🤖 Gemini: ¡Hola! Soy Gemini. Escribe 'salir' para terminar.\n")

    while True:
        user_input = input("Tú: ")

        if user_input.lower().strip() in ["salir", "exit", "quit"]:
            print("🤖 Gemini: ¡Hasta luego!")
            break

        try:
            response = chat.send_message(user_input)
            print(f"🤖 Gemini: {response.text}\n")
        except Exception as e:
            print(f"❌ Error al enviar mensaje: {e}\n")

if __name__ == "__main__":
    chat_consola()
