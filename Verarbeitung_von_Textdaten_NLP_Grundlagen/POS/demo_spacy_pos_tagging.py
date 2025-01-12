# Importieren der notwendigen Bibliotheken
import spacy

# SpaCy ist eine natürliche Sprachverarbeitungsbibliothek in Python
# Zuerst müssen wir das Sprachmodell laden, das wir für das POS-Tagging verwenden wollen.
# In diesem Beispiel verwenden wir das deutsche Sprachmodell 'de_core_news_sm'.

# Wenn das Modell noch nicht installiert ist, können Sie es mit folgendem Befehl installieren:
# !python -m spacy download de_core_news_sm

# Laden des deutschen SpaCy Modells
nlp = spacy.load("de_core_news_sm")

# Funktion zum Durchführen von POS-Tagging
def pos_tagging(text):
    """
    Diese Funktion nimmt einen Text als Eingabe und gibt die POS-Tagging-Informationen zurück.
    
    Parameters:
    text (str): Der Eingabetext, der analysiert werden soll.
    
    Returns:
    list: Eine Liste von Tupeln, die jedes Token und dessen POS-Tag repräsentieren.
    """
    # Verarbeiten des Textes mit dem SpaCy NLP Modell
    doc = nlp(text)
    
    # Erstellen einer Liste der Token und deren POS-Tags
    pos_tags = [(token.text, token.pos_) for token in doc]
    
    return pos_tags

# Beispieltext
text = "Die Katze sitzt auf der Matte."

# Aufruf der Funktion und Speichern der Ergebnisse
pos_tags_result = pos_tagging(text)

# Ausgabe der POS-Tagging Ergebnisse
print("POS-Tagging Ergebnis:")
for token, pos in pos_tags_result:
    print(f"{token}: {pos}")

# Anpassungen für Mehrsprachigkeit
# SpaCy unterstützt mehrere Sprachen. Sie können das Sprachmodell ändern,
# indem Sie einfach ein anderes Modell laden, z.B. 'en_core_web_sm' für Englisch.

# Beispiel für englischen Text
# nlp_en = spacy.load("en_core_web_sm")
# text_en = "The cat is sitting on the mat."
# pos_tags_result_en = pos_tagging(text_en)
# print("\nPOS-Tagging Ergebnis auf Englisch:")
# for token, pos in pos_tags_result_en:
#     print(f"{token}: {pos}")
