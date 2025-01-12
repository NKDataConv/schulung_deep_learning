# Importieren der notwendigen Bibliotheken
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
import pandas as pd

# Überprüfen, ob eine GPU verfügbar ist
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Verwendetes Gerät: {device}')

# Definieren der Hyperparameter und Modellname
model_name = "distilbert-base-uncased"
checkpoint_path = "./model_checkpoint"  # Pfad zum Checkpoint-Verzeichnis

# Schritt 1: Laden des Tokenizers und des Modells
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Funktion zum Laden eines Modells aus einem Checkpoint
def load_model(checkpoint_path):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    return model

# Schritt 2: Vorbereitung der Eingabedaten für die Inferenz
def prepare_data(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

# Schritt 3: Inferenzfunktion für das geladene Modell
def inference(model, texts):
    inputs = prepare_data(texts)
    with torch.no_grad():  # Deaktivieren der Gradientenberechnung für den Inferenzprozess
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions

# Schritt 4: Hauptfunktion für die Demo
def main():
    # Laden des Modells aus dem Checkpoint
    print("Lade Modell aus Checkpoint...")
    model = load_model(checkpoint_path)
    
    # Beispieltexte zur Inferenz
    example_texts = [
        "I love programming!",
        "I'm not a fan of bugs in my code.",
        "Deep learning is an exciting field."
    ]
    
    # Durchführung der Inferenz
    print("Durchführung der Inferenz...")
    predictions = inference(model, example_texts)
    
    # Ausgabe der Vorhersagen
    for text, pred in zip(example_texts, predictions):
        print(f'Text: "{text}" - Vorhersage: {pred.item()}')

# Schritt 5: Ausführen der Hauptfunktion
if __name__ == "__main__":
    main()

# **Erläuterungen und wichtige Punkte:**
#
# - **Checkpointing:** In diesem Skript wird das Konzept des Checkpointings durch die Verwendung eines gespeicherten Modells anhand des `checkpoint_path`-Wertes verdeutlicht. Vorhandene Modellversionen werden geladen, was wichtig ist, um den Zustand eines Modells während des Trainings oder nach dem Training zu erhalten.
#
# - **Laden und Verwenden des Modells:** Die Funktion `load_model` zeigt, wie man ein vorbereitetes Modell aus einem Checkpoint lädt. Dies ist wichtig, um auf frühere Trainingsversionen zurückzugreifen.
#
# - **Inferenz:** Die Funktion `inference` demonstriert, wie das geladene Modell für Vorhersagen auf neuen Eingabedaten verwendet werden kann.
#
# - **Tokenisierung:** Die Verwendung des `tokenizer` ist entscheidend, um sicherzustellen, dass die Texte in das geeignete Format für das Modell konvertiert werden.
#
# - **Zukunftssichere Entwicklung:** Das Skript ist so strukturiert, dass es leicht verständlich ist und sich leicht erweitern lässt, was für eine Schulung nützlich ist. Die Modularität des Codes fördert ein besseres Verständnis der Funktionsweise von Modellen und deren Inferenz und ermöglicht das Hinzufügen weiterer Funktionen, wie z. B. das Speichern von Checkpoints während des Trainings.