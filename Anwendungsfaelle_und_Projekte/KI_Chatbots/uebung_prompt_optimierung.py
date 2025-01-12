"""
Aufgabenstellung:
In dieser Übung werden wir die Anpassung der Prompt-Strategie für spezifische Dialogszenarien untersuchen.
Ziel der Übung ist es, die Effektivität von KI-Chatbots mit vortrainierten Transformers für unterschiedliche
Dialogszenarien zu verbessern. Dazu gehört, dass Sie spezifische Anfrage-Prompts gestalten, um die Antworten
des Transformers in die gewünschte Richtung zu lenken.

Schritte:
1. Verwenden Sie die OpenAI GPT-Transformer-Bibliothek, um ein vortrainiertes Sprachmodell zu laden.
2. Definieren Sie verschiedene spezifische Dialogszenarien, z.B. Kundenservice, Technischer Support und Small Talk.
3. Erstellen Sie für jedes Szenario spezifische Prompts, um die Effektivität des Modells zu testen.
4. Passen Sie die Prompts an und analysieren Sie die Ausgabe des Modells.
5. Optimieren Sie die Prompts basierend auf der Qualität der Antworten, um die gewünschten Dialoge zu erhalten.

Hinweis: Stellen Sie sicher, dass Sie die OpenAI Python-Bibliothek installiert haben (`pip install openai`).
Verwenden Sie Ihren OpenAI-API-Schlüssel.

Beachten Sie, dass die Ausführung des Skripts API-Anfragen an das OpenAI-Modell stellt und dabei Kosten
anfallen können.
"""

import openai

# Stellen Sie sicher, dass Ihr OpenAI-API-Schlüssel hier konfiguriert ist
openai.api_key = 'your-api-key-here'

# Funktion zur Abfrage des Modells
def ask_gpt(prompt):
    """
    Diese Funktion sendet eine Anfrage an das GPT-Modell und gibt die Antwort zurück.
    
    :param prompt: Der Anfrage-Prompt, der an das Modell gesendet werden soll.
    :return: Antwort des Modells als Text.
    """
    # Senden einer Anfrage an das OpenAI Modell
    response = openai.Completion.create(
        engine="text-davinci-003",  # Verwenden Sie das entsprechende GPT-Modell
        prompt=prompt,
        max_tokens=100,  # Maximale Länge der Antwort
        n=1,  # Anzahl der zu erhaltenden Antwortmöglichkeiten
        stop=None,  # Ende der Antwort (keine spezifischen Stoppzeichen)
        temperature=0.7  # Kreativität der Antwort
    )
    # Geben Sie den generierten Text zurück
    return response.choices[0].text.strip()

# Definieren spezifischer Dialogszenarien
dialog_scenarios = {
    'kundenservice': "Ein Kunde ruft an und beschwert sich über eine verspätete Lieferung. Wie antworten Sie?",
    'technischer_support': "Ein Benutzer meldet ein Problem beim Einloggen in sein Konto. Wie unterstützen Sie ihn?",
    'small_talk': "Sie treffen zufällig jemanden auf einer Konferenz. Wie beginnen Sie das Gespräch?"
}

# Verarbeiten und Anpassen von Prompts für jede Situation
for scenario_name, prompt in dialog_scenarios.items():
    print(f"\nSzenario: {scenario_name}")
    print("===================================")
    # Anpassen und analysieren der Prompts
    response = ask_gpt(prompt)
    print(f"GPT Antwort: {response}")
    
    # Hier kann eine Optimierung erfolgen, indem man das prompt anpasst basierend auf
    # der Analyse der Modellantworten. Beispiel:
    # Einfache Analyse: Wenn die Antwort nicht detailreich genug ist, könnte man versuchen,
    # mehr Kontext hinzuzufügen oder konkretisierende Fragen zu stellen.
    # prompt_anpassung = prompt + " Bitte geben Sie mir mehr Details."
    # response_optimiert = ask_gpt(prompt_anpassung)
    # print(f"Optimierte GPT Antwort: {response_optimiert}")
