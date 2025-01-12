# Importieren der erforderlichen Bibliotheken
import streamlit as st
from transformers import pipeline

# Hier wird ein vortrainiertes Sprachmodell geladen.
# In diesem Beispiel verwenden wir das 'conversational' Modell der Hugging Face Transformers Library
@st.cache_resource
def load_chatbot():
    chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")
    return chatbot

# Funktion zur Verarbeitung der Benutzereingaben und zur Generierung von Antworten
def generate_response(user_input):
    # Erstellen einer Konversation
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Speicherung der Benutzereingabe in der Konversation
    user_input_message = {
        "role": "user",
        "content": user_input
    }
    st.session_state.history.append(user_input_message)

    # Generieren der Antwort des Chatbots
    response = chatbot(st.session_state.history)

    # Die Antwort des Chatbots wird zur Konversation hinzugefügt
    st.session_state.history.append(response)

    # Rückgabe der Antwort als Text
    return response.generated_responses[-1]

# Hauptfunktion für die Streamlit-App
def main():
    st.title("KI-Chatbot mit vortrainierten Transformern")
    st.write("Willkommen! Stellen Sie mir eine Frage oder äußern Sie einen Kommentar!")

    # Eingabefeld für den Benutzer
    user_input = st.text_input("Ihre Eingabe:", "")

    # Wenn der Benutzer eine Eingabe macht und die Eingabe nicht leer ist
    if user_input:
        # Generieren der Antwort des Chatbots
        chatbot_response = generate_response(user_input)

        # Anzeige der generierten Antwort
        st.write("💬 Chatbot: ", chatbot_response)

        # Optional: Verlauf der Konversation anzeigen
        if st.checkbox("Verlauf anzeigen"):
            st.write("### Verlauf der Konversation:")
            for message in st.session_state.history:
                if message["role"] == "user":
                    st.write("🔹 Benutzer: ", message["content"])
                else:
                    st.write("🔹 Chatbot: ", message['content'])

# Dies ist der Einstiegspunkt für die Streamlit-Anwendung
if __name__ == "__main__":
    # Laden des Chatbots
    chatbot = load_chatbot()
    # Ausführen der Hauptfunktion
    main()


### Erläuterungen:
# - **Libraries**: Wir verwenden `streamlit` für die Weboberfläche und `transformers` von Hugging Face für den Zugriff auf vortrainierte Modelle.
# - **Caching**: Mit `@st.cache_resource` sorgen wir dafür, dass das Chatbot-Modell nur einmal geladen wird, um die Leistung zu verbessern.
# - **Session State**: `st.session_state` wird verwendet, um den Verlauf der Konversation beizubehalten, sodass der Chatbot den Kontext der Diskussion verfolgen kann.
# - **Textinput**: Das Benutzertextfeld ermöglicht es den Benutzern, ihre Eingaben zu tätigen.
# - **Response Generation**: Das Skript verarbeitet die Benutzereingabe und gibt eine Antwort zurück, die im Webinterface angezeigt wird.
# - **Conversation History**: Die Möglichkeit, den Verlauf der Konversation anzuzeigen, bietet Einblick in die Interaktionen zwischen Benutzer und Chatbot.

### Ausführung der Streamlit-Anwendung:
# Starten Sie die Anwendung mit folgendem Befehl im Terminal:
# ```bash
# streamlit run <dateiname>.py
# ```
