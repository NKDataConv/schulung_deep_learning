# Importiere notwendige Bibliotheken
import torch
from transformers import MarianMTModel, MarianTokenizer

# Überprüfen, ob eine GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definiere das Übersetzungsmodell und den Tokenizer
# In diesem Beispiel verwenden wir ein vortrainiertes Modell für die Übersetzung von Englisch nach Deutsch
model_name = "Helsinki-NLP/opus-mt-en-de"

# Lade den Tokenizer und das Modell
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

def translate(text, max_length=40):
    """
    Übersetzt den gegebenen Text von Englisch nach Deutsch.
    
    :param text: Der zu übersetzende Text (String)
    :param max_length: Maximale Länge der Übersetzung (Integer)
    :return: Übersetzter Text (String)
    """
    # Tokenisierung des Eingangstextes
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Generiere Übersetzung mit dem Modell
    translated = model.generate(**inputs, max_length=max_length)
    
    # Dekodieren der generierten Tokens zurück zu Text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return translated_text

# Beispieltext für die Übersetzung
example_text = "Hello, how are you doing today?"

# Ausgabe des ursprünglichen Textes
print("Original Text: ", example_text)

# Führe die Übersetzung durch
translated_text = translate(example_text)

# Ausgabe des übersetzten Textes
print("Translated Text: ", translated_text)
