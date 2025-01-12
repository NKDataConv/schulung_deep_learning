# Importieren der erforderlichen Bibliotheken
# Wir verwenden die Hugging Face Transformers-Bibliothek für vortrainierte Modelle
from transformers import pipeline

# Definieren Sie die Funktion zur Generierung einer Textzusammenfassung
def generative_summary(text, model_name='facebook/bart-large-cnn'):
    """
    Generiert eine zusammenfassende Version des eingegebenen Textes 
    mithilfe eines vortrainierten Modells.

    :param text: Der einzugeben Text, der zusammengefasst werden soll.
    :param model_name: Der Name des vortrainierten Modells.
    :return: Generierte Zusammenfassung des Textes.
    """
    
    # Erstellen eines Summarization-Pipelines mit dem angegebenen Modell
    summarizer = pipeline("summarization", model=model_name)

    # Durchführung der Zusammenfassung, wobei die Länge der Zusammenfassung festgelegt werden kann
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)

    # Extrahieren der tatsächlichen Zusammenfassung aus dem Ausgabeformat
    return summary[0]['summary_text']

# Textbeispiel für die Zusammenfassung
example_text = """
Deep Learning ist ein Teilbereich des maschinellen Lernens und basiert auf künstlichen neuronalen Netzwerken. 
Es hat in den letzten Jahren durch die Verfügbarkeit großer Datenmengen und leistungsfähiger Computer 
eine immense Pionierleistung erbracht. Anwendungen reichen von Bild- und Spracherkennung bis hin zu 
Textverarbeitung. Die Grundlagen von Deep Learning umfassen neuronale Netzwerke, Aktivierungsfunktionen 
sowie Verlustfunktionen. Bei der Verarbeitung von Textdaten ist es wichtig, den Text in ein Format zu 
überführen, das für maschinelles Lernen geeignet ist. Techniken des Natural Language Processing (NLP) 
sind hierbei entscheidend. Transformer-Modelle stellen einen bedeutenden Fortschritt dar und ermöglichen 
die gleichzeitige Verarbeitung von Eingabesequenzen, wodurch das Lernen von semantischen Beziehung verbessert wird.
"""

# Aufrufen der Funktion zur Erstellung einer generativen Zusammenfassung
if __name__ == "__main__":
    # Generieren der Zusammenfassung für das Beispieltext
    summary_text = generative_summary(example_text)
    
    # Ausgabe der generierten Zusammenfassung
    print("Originaltext:\n", example_text)
    print("\nGenerierte Zusammenfassung:\n", summary_text)
