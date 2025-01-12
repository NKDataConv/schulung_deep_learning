# Importieren der notwendigen Bibliotheken
from transformers import pipeline

# Named Entity Recognition (NER) ist eine Technik im Bereich der natürlichen Sprache,
# die es ermöglicht, spezifische Entitäten in einem gegebenen Text zu identifizieren.
# Typische Entitäten sind Personen, Organisationen, Orte, Datumsangaben etc.

# In dieser Demo verwenden wir die Hugging Face Transformers-Bibliothek,
# um eine vortrainierte NER-Pipeline anzuwenden.

def main():
    # Initialisierung der NER-Pipeline mit einem vortrainierten Modell
    ner_pipeline = pipeline("ner", aggregation_strategy="simple")

    # Der Text, aus dem wir Named Entities extrahieren wollen
    text = ("Barack Obama wurde am 4. August 1961 in Honolulu, Hawaii geboren. "
            "Er war der 44. Präsident der Vereinigten Staaten und diente von 2009 bis 2017.")

    # Anzeige des Originaltexts
    print("Originaltext:")
    print(text)
    print("\nErkannte Benannte Entitäten:\n")

    # Anwendung der NER-Pipeline auf den Text
    entities = ner_pipeline(text)

    # Ausgabe der erkannten Entitäten
    for entity in entities:
        # Jede Entität hat eine Kategorie, einen Text und eine Wahrscheinlichkeit
        print(f"Entität: {entity['word']}, "
              f"Kategorie: {entity['entity_group']}, "
              f"Wahrscheinlichkeit: {entity['score']:.2f}, "
              f"Start: {entity['start']}, "
              f"Ende: {entity['end']}")

# Überprüfung, ob das Skript direkt ausgeführt wird
if __name__ == "__main__":
    main()


### Erläuterungen:
# 1. **Importieren der Bibliotheken**: Das Skript verwendet die `pipeline`-Funktion aus der Hugging Face Transformers-Bibliothek, die eine einfache Schnittstelle für verschiedene NLP-Aufgaben bietet.
#
# 2. **Initialisierung der Pipeline**: Die NER-Pipeline wird mit dem Modell zum Erkennen benannter Entitäten konfiguriert. Hier wird `aggregation_strategy="simple"` verwendet, um ähnliche Entitäten zusammenzufassen.
#
# 3. **Text zur Analyse**: Ein Beispieltext wird definiert, der Informationen über Barack Obama enthält.
#
# 4. **Erkennung der Entitäten**: Die Pipeline wird auf den definierten Text angewendet, und die erkannten Entitäten werden in einer strukturierten Form ausgegeben.
#
# 5. **Ergebnisse**: Für jede erkannte Entität werden die Details (Text, Kategorie, Wahrscheinlichkeit, Start- und Endposition) ausgegeben.
