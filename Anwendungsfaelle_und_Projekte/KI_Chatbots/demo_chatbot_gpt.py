# Importiere die benötigten Bibliotheken
import openai

# Setze deinen OpenAI API-Schlüssel hier ein
# Du erhältst diesen Schlüssel, wenn du ein OpenAI-Konto erstellst und API-Zugriff beantragst.
openai.api_key = 'DEIN_API_SCHLÜSSEL_HIER'

def gpt_chat_session():
    """
    Diese Funktion ermöglicht es, eine einfache Chat-Session mit einem 
    GPT-basierten Modell zu starten. Der Benutzer kann Fragen stellen, 
    und das Modell wird darauf basierend Antworten generieren.
    """

    print("Willkommen zur einfachen Chat-Session mit einem GPT-basierten Modell!")
    print("Tippe 'exit' zum Beenden der Session.\n")

    # Eine Endlosschleife für die Chat-Session
    while True:
        # Nimmt die Eingabe vom Benutzer
        user_input = input("Du: ")
        
        # Überprüft, ob der Benutzer die Session beenden möchte
        if user_input.lower() == 'exit':
            print("Chat-Session beendet.")
            break

        # Generiere eine Antwort vom OpenAI GPT-3 Modell
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Verwende das GPT-3.5 Turbo Modell
            messages=[
                {"role": "user", "content": user_input}  # Übergebe die Benutzereingabe
            ]
        )

        # Hole die Antwort aus der API-Antwort
        answer = response.choices[0].message['content']
        print("GPT: " + answer)  # Gibt die Antwort aus

# Stelle sicher, dass das Skript nicht direkt ausgeführt wird, sondern nur importiert
if __name__ == "__main__":
    gpt_chat_session()

### Erklärung des Codes:
# 1. **Importieren der Bibliotheken**: Wir importieren die OpenAI-Bibliothek, um auf die GPT-API zuzugreifen.
#
# 2. **API-Schlüssel**: Der `openai.api_key` wird gesetzt, um sich mit der OpenAI-API zu authentifizieren. Diesen Schlüssel erhältst du nach der Registrierung bei OpenAI.
#
# 3. **Chat-Session Funktion**: Die Funktion `gpt_chat_session` startet eine Endlosschleife, in der der Benutzer Fragen stellen kann, bis er 'exit' eingibt.
#
# 4. **Benutzereingabe**: Die Eingabe des Nutzers wird erfasst und auf 'exit' geprüft, um die Sitzung zu beenden.
#
# 5. **GPT-3 API Aufruf**: Der Benutzerinput wird an die OpenAI Chat-Completion-API gesendet, die eine Antwort basierend auf dem GPT-3.5-Turbo Modell generiert.
#
# 6. **Ausgabe der Antwort**: Die Antwort des Modells wird auf der Konsole ausgegeben.
