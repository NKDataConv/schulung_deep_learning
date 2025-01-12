# Importieren der notwendigen Bibliotheken
from fastapi import FastAPI
from pydantic import BaseModel
import joblib  # Zum Laden des Modells
import numpy as np

# Schritt 1: Erstellen einer FastAPI-Anwendung
app = FastAPI()

# Schritt 2: Modell laden
# Hier laden wir ein vortrainiertes Modell, das wir zuvor gespeichert haben.
# Stellen Sie sicher, dass das Modell im gleichen Verzeichnis gespeichert ist.
model_path = "path/to/your/model.joblib"  # Pfad zum vortrainierten Modell
model = joblib.load(model_path)

# Schritt 3: Definieren von Datenmodellen
# Wir definieren ein Pydantic-Datenmodell für die Eingaben, die wir vom Benutzer erwarten.
class TextInput(BaseModel):
    text: str  # Die Eingabe, die das Modell benötigt

# Schritt 4: Erstellen einer Endpoint-Funktion
@app.post("/predict")  # Die URL, um Vorhersagen zu machen
async def predict(input: TextInput):
    """
    Diese Funktion akzeptiert Textinput,
    verwendet das geladene Modell zur Vorhersage
    und gibt das Ergebnis zurück.
    """
    # Schritt 4a: Vorverarbeitung des Textes
    # Hier sollten Sie Ihren Vorverarbeitungscode hinzufügen, z.B. Tokenisierung,
    # Stopwortentfernung, Vektorisierung usw.
    processed_text = preprocess_text(input.text)

    # Schritt 4b: Modellvorhersage
    prediction = model.predict([processed_text])  # Das Modell erwartet ein 2D-Array

    # Schritt 4c: Rückgabe der Vorhersage
    return {"prediction": prediction[0]}  # Wir geben nur den ersten Wert zurück

# Schritt 5: Vorverarbeitungsfunktion (Platzhalter)
def preprocess_text(text: str):
    """
    Diese Funktion dient zur Vorverarbeitung des eingegebenen Textes.
    Hier können Sie Schritte wie Tokenisierung, Stopwortentfernung und Vektorisierung durchführen.
    """
    # Beispiel für einfache Vorverarbeitung. Passen Sie diese an Ihre Bedürfnisse an.
    text = text.lower()  # In Kleinbuchstaben umwandeln
    # Weitere Schritte wie Entfernen von Sonderzeichen, Tokenisierung usw. können hier hinzugefügt werden
    return text  # Rückgabe des vorverarbeiteten Textes

# Schritt 6: Starten des FastAPI-Servers
# Um den Server zu starten, führen Sie das folgende Kommando im Terminal aus:
# uvicorn your_script_name:app --reload
# Ersetzen Sie 'your_script_name' durch den tatsächlichen Namen Ihres Python-Skripts.


### Wichtige Hinweise:
# - **Modell speichern:** Stellen Sie sicher, dass Ihr Modell mit `joblib` oder einem anderen geeigneten Format gespeichert wurde, bevor Sie dieses Skript ausführen.
# - **Modell-anpassung:** Die Vorverarbeitungsfunktion `preprocess_text` sollte angepasst werden, um alle spezifischen Anforderungen hinsichtlich der Eingabedaten zu erfüllen. Hier können weitere Bibliotheken wie `nltk`, `spaCy` oder `scikit-learn` verwendet werden, um die Vorverarbeitungslogik zu verfeinern.
# - **Deployment:** Um das Modell zu deployen, benötigen Sie `Uvicorn`, um den FastAPI-Server lokal zu starten. Installieren Sie es gegebenenfalls mit `pip install uvicorn`.
# - **Tests:** Testen Sie Ihren API-Endpunkt mit Tools wie `Postman` oder `curl`, um sicherzustellen, dass alles wie erwartet funktioniert.
