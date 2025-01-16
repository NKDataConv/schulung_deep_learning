# Importieren der notwendigen Bibliotheken
from transformers import pipeline

# In dieser Demo verwenden wir die Hugging Face Transformers-Bibliothek,
# um eine vortrainierte NER-Pipeline anzuwenden.

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
