"""
Aufgabenstellung:

In dieser Übung geht es darum, zwei unterschiedliche Modelle zur Textzusammenfassung
zu verwenden und die generierten Zusammenfassungen miteinander zu vergleichen. 

1. Wähle zwei vortrainierte Modelle für die Textzusammenfassung: ein extraktives 
   Modell (z.B. Bart von Hugging Face) und ein generatives Modell (z.B. T5 von Hugging Face).

2. Lade einen Beispieltext, der zusammengefasst werden soll. 

3. Verwende beide Modelle, um jeweils eine Zusammenfassung des Textes zu erzeugen.

4. Vergleiche die generierten Zusammenfassungen der beiden Modelle manuell und 
   analysiere die Unterschiede in Bezug auf Genauigkeit, Kohärenz und Lesbarkeit.

5. Dokumentiere deine Beobachtungen und diskutieren Sie die Vor- und Nachteile 
   von extraktiven und generativen Ansätzen in einer Schulungsdiskussion.

"""

import torch
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration

# Überprüfe, ob ein GPU verfügbar ist, um den Prozess zu beschleunigen
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Beispieltext, der zusammengefasst werden soll
example_text = """
Künstliche Intelligenz (KI) ist ein Bereich der Informatik, der darauf abzielt, Maschinen zu entwickeln, 
die menschliches Denken und Verhalten nachahmen können. Fortschritte in diesem Bereich führen zu Entwicklungen 
von Systemen, die in der Lage sind, komplexe Probleme zu lösen und Aufgaben eigenständig zu bewältigen.
"""


# Funktion, um Text mit einem Modell zusammenzufassen
def summarize_with_model(model_type, model_name, text):
    """
    Param: model_type - Der Typ des Modells ('bart' oder 't5')
    Param: model_name - Der Name des vortrainierten Modells
    Param: text - Der Text, der zusammengefasst werden soll

    Retour: Zusammenfassung des Textes
    """
    # Initialisiere den entsprechenden Tokenizer und das Modell
    if model_type == 'bart':
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    elif model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    else:
        raise ValueError("Ungültiger Modelltyp. Wählen Sie 'bart' oder 't5'.")

    # Tokenisisere den Eingangstext und verschiebe ihn auf das verwendete Gerät (CPU oder GPU)
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    # Generiere die Zusammenfassung
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


# Modellnamen für beide Ansätze
bart_model_name = 'facebook/bart-large-cnn'
t5_model_name = 't5-base'

# Generiere die Zusammenfassungen mit beiden Modellen
bart_summary = summarize_with_model('bart', bart_model_name, example_text)
t5_summary = summarize_with_model('t5', t5_model_name, example_text)

# Ausgabe der Zusammenfassungen
print("Zusammenfassung mit BART:")
print(bart_summary)
print("\nZusammenfassung mit T5:")
print(t5_summary)
