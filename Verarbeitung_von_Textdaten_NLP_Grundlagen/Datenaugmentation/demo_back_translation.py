# Importiere die erforderlichen Bibliotheken
from transformers import MarianMTModel, MarianTokenizer

# Funktion zur Rückübersetzung (Back Translation)
def back_translate(text, src_language, target_language):
    # Wähle das MarianMT Modell basierend auf den Quell- und Zielsprachen
    model_name = f'Helsinki-NLP/opus-mt-{src_language}-{target_language}'
    
    # Lade den Tokenizer und das Modell für die Übersetzungen
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Übersetze den Text in die Zielsprache
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True, truncation=True))
    # Dekodiere die übersetzte Antwort
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    # Übersetze den Text zurück zur Ausgangssprache
    model_name_back = f'Helsinki-NLP/opus-mt-{target_language}-{src_language}'
    tokenizer_back = MarianTokenizer.from_pretrained(model_name_back)
    model_back = MarianMTModel.from_pretrained(model_name_back)
    
    back_translated = model_back.generate(**tokenizer_back(translated_text, return_tensors="pt", padding=True, truncation=True))
    # Dekodiere die zurückübersetzte Antwort
    back_translated_text = tokenizer_back.decode(back_translated[0], skip_special_tokens=True)
    
    return back_translated_text

# Beispieltext zur Verwendung in der Back-Translation
source_text = "Deep Learning ist ein Teilbereich des maschinellen Lernens, der auf künstlichen neuronalen Netzen basiert."

# Sprachen definieren
source_language = 'de'  # Deutsch
target_language = 'en'   # Englisch

# Führe die Rückübersetzung durch
augmented_text = back_translate(source_text, source_language, target_language)

# Zeige die Ergebnisse
print(f"Ursprünglicher Text: {source_text}")
print(f"Augmentierter Text durch Back-Translation: {augmented_text}")

# Funktion zur Durchführung von mehreren Back-Translations mit Variation
def multiple_back_translations(text, src_language, target_language, num_translations=5):
    results = set()  # Verwende ein Set, um Duplikate zu vermeiden
    for _ in range(num_translations):
        augmented = back_translate(text, src_language, target_language)
        results.add(augmented)  # Füge das Ergebnis zum Set hinzu
    return list(results)

# Mehrere Rückübersetzungen durchführen
augmented_texts = multiple_back_translations(source_text, source_language, target_language)

print("\nMehrere Augmentierungen durch Back-Translation:")
for i, augmented in enumerate(augmented_texts, start=1):
    print(f"{i}: {augmented}")
