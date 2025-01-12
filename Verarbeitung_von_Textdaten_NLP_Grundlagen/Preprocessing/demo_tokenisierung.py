# Importieren der erforderlichen Bibliotheken
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Sicherstellen, dass die erforderlichen NLTK-Ressourcen vorhanden sind
nltk.download('punkt')          # Für die Tokenisierung
nltk.download('stopwords')      # Für die Stopwörter
nltk.download('wordnet')        # Für das Lemmatisieren
nltk.download('punkt_tab')      # Für die Vektorisierung

# Beispieltext, der verarbeitet wird
text_data = [
    "Deep learning is a subset of machine learning.",
    "Natural language processing (NLP) enables machines to understand human language.",
    "Tokenization is the process of converting text into tokens."
]

# Schritt 1: Textbereinigung
def clean_text(text):
    """
    Bereinigt den Text durch Entfernen von Sonderzeichen und Kleinbuchstaben.
    :param text: Eingabetext
    :return: Bereinigter Text
    """
    # Entfernen von HTML-Tags, falls vorhanden
    text = re.sub(r'<.*?>', '', text)
    # Entfernen von nicht-alphanumerischen Zeichen
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Konvertieren in Kleinbuchstaben
    text = text.lower()
    return text

# Schritt 2: Tokenisierung
def tokenize_text(text):
    """
    Tokenisiert den Text in einzelne Wörter.
    :param text: Bereinigter Text
    :return: Liste von Tokens
    """
    return word_tokenize(text)

# Schritt 3: Entfernen von Stopwörtern
def remove_stopwords(tokens):
    """
    Entfernt Stopwörter aus der Liste von Tokens.
    :param tokens: Liste von Tokens
    :return: Liste von Tokens ohne Stopwörter
    """
    stop_words = set(stopwords.words('english'))  # Englische Stopwörter
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

# Schritt 4: Lemmatisierung
def lemmatize_tokens(tokens):
    """
    Lemmatisiert die Tokens auf ihre Grundform.
    :param tokens: Liste von Tokens
    :return: Liste von lemmatisierten Tokens
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# Schritt 5: Textvorverarbeitung und Hauptfunktion
def preprocess_text(text):
    """
    Vorverarbeitung: Bereinigung, Tokenisierung, Entfernen von Stopwörtern und Lemmatisierung.
    :param text: Eingabetext
    :return: Liste von lemmatisierten Tokens
    """
    cleaned_text = clean_text(text)  # Bereinigung
    tokens = tokenize_text(cleaned_text)  # Tokenisierung
    tokens_no_stopwords = remove_stopwords(tokens)  # Stopwörter entfernen
    lemmatized_tokens = lemmatize_tokens(tokens_no_stopwords)  # Lemmatisierung
    return lemmatized_tokens

# Anwendung der Vorverarbeitung auf die Beispieldaten
preprocessed_data = [preprocess_text(text) for text in text_data]

# Ausgabe der verarbeiteten Textdaten
print("Bearbeitete Textdaten:")
for i, text in enumerate(preprocessed_data):
    print(f"Text {i + 1}: {text}")

# Schritt 6: Vektorisierung (optional)
def vectorize_text(preprocessed_data):
    """
    Vektorisiert die lemmatisierten Tokens in eine Bag-of-Words-Darstellung.
    :param preprocessed_data: Liste von lemmatisierten Tokens
    :return: Bag-of-Words-Matrix
    """
    # Join Tokens für Vektorisierung
    text_for_vectorization = [' '.join(tokens) for tokens in preprocessed_data]
    vectorizer = CountVectorizer()
    boW_matrix = vectorizer.fit_transform(text_for_vectorization)
    return boW_matrix, vectorizer.get_feature_names_out()

# Vektorisierung der verarbeiteten Textdaten
boW_matrix, feature_names = vectorize_text(preprocessed_data)

# Ausgabe der Bag-of-Words-Matrix
print("\nBag-of-Words-Matrix:")
print(boW_matrix.toarray())
print("Feature-Namen:", feature_names)
