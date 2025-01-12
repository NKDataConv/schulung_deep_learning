# Import der notwendigen Bibliotheken
import prodigy
from prodigy.components.preprocessors import TextInput, add_tokens
from prodigy.models.ner import EntityRecognizer
from pathlib import Path
import spacy

# Initialisierung eines leeren NER Modells
nlp = spacy.blank("en")  # Erstelle ein leeres englisches Spacy-Modell

# Definiere die NER-Labels, die annotiert werden sollen
labels = ["PERSON", "ORG", "GPE", "LOC", "DATE"]

# Funktion, um eine Prodigy-Annotation zu starten
def start_prodigy_ner():
    """Startet eine Prodigy-Sitzung zur Annotation für NER."""
    # Prodigy-Befehl, um ein NER-Tagging-Projekt mit Active Learning zu starten
    prodigy.serve(
        "ner.teach",  # Prodigy-Recipe für das NER-Training
        "my_dataset",  # Name des Datensatzes, der gespeichert wird
        "path/to/your/textfile.txt",  # Pfad zu den zu annotierenden Texten
        model=nlp,  # Das NER Modell
        labels=labels,  # Die definierten Labels
        view_id="ner",  # Die Ansicht für NER
        config={"global": {"batch_size": 8}},  # Konfiguration der Batch-Größe
    )

# Funktion zur Verarbeitung der Annotationsergebnisse
def process_annotations(dataset):
    """Verarbeitet die Annotationen aus dem Prodigy-Datensatz."""
    # Lese Annotationen aus dem Prodigy-Datensatz
    annotations = prodigy.load_jsonl(f"{dataset}.jsonl")

    for annotation in annotations:
        if annotation.get("accept") == True:
            # Zugang zu den annotierten Inhalten
            text = annotation["text"]
            # Zugang zu den annotierten Entitäten
            entities = annotation["spans"]

            # Beispiel: Ausgabe der akzeptierten Entitäten
            print(f"Akzeptierte Entität: {text}")
            for ent in entities:
                print(f"  - {ent['label']} von {ent['start']} bis {ent['end']}")

# Funktion, um ein neues NER-Modell mit den annotierten Daten zu trainieren
def train_ner_model(dataset):
    """Trainiert das NER-Modell mit den gesammelten Daten aus Prodigy."""
    # Lade die Prodigy-Daten
    annotations_data = prodigy.load_jsonl(f"{dataset}.jsonl")

    # Extrahiere die Texte und die zugehörigen Entitäten für das Training
    training_data = []
    for annotation in annotations_data:
        if annotation.get("accept"):
            text = annotation["text"]
            entities = [(ent["start"], ent["end"], ent["label"]) for ent in annotation["spans"]]
            training_data.append((text, {"entities": entities}))

    # Training des Modells
    for text, annotations in training_data:
        # Trainiere das NER-Modell mit den annotierten Daten
        print(f"Trainiere mit dem Text: {text}")
        nlp.update([text], [annotations])

# Hauptfunktion zur Ausführung des Skripts
def main():
    """Hauptfunktion zum Ausführen der NER-Demo mit Prodigy."""
    # Schritt 1: Starte die Prodigy-Annotation für NER
    print("Starte die Prodigy-NER Annotation...")
    start_prodigy_ner()

    # Schritt 2: Nach Beendigung der Annotation die Ergebnisse verarbeiten
    print("Verarbeite die Annotationsergebnisse...")
    process_annotations("my_dataset")  # Geben Sie hier den Namen des Prodigy-Datensatzes an

    # Schritt 3: Trainiere das NER-Modell mit den annotierten Daten
    print("Trainiere das NER-Modell...")
    train_ner_model("my_dataset")

if __name__ == "__main__":
    main()


### Erklärung:

# - **Prozessaufteilung:** Das Skript ist in verschiedene Funktionen unterteilt, um die Übersichtlichkeit zu gewährleisten.
# - **Imports:** Die notwendigen Bibliotheken wie `prodigy`, `spacy`, und `pathlib` werden importiert.
# - **Label-/Modelldefinition:** Es wird ein leeres Spacy-Modell erstellt, und die zu annotierenden Entitäten werden definiert.
# - **Prodigy-Annotation:** Die Funktion `start_prodigy_ner` startet eine Prodigy-Sitzung, um Textdaten zu annotieren.
# - **Annotation Prozess:** Die Funktion `process_annotations` verarbeitet die annotierten Daten, um akzeptierte Entitäten zu extrahieren.
# - **Trainingsprozess:** Das NER-Modell wird mithilfe der annotierten Daten in der Funktion `train_ner_model` trainiert.
# - **Hauptfunktion:** Die `main`-Funktion steuert den gesamten Ablauf von Annotation bis Training.
