# Importieren der erforderlichen Bibliotheken
import numpy as np
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import logging

# NLTK-Bibliotheken für die Tokenisierung und andere natürliche Sprachverarbeitungsfunktionen verwenden
nltk.download('punkt')  # Herunterladen des Punktes Tokenizers

# Aktiviere Logging, um den Fortschritt der Word2Vec-Trainingsausgaben anzuzeigen
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Beispielkorpus - eine kleine Sammlung von Texten, denselben Text mehrmals wiederholt, um mehr Daten zu haben
corpus = [
    "Deep learning is a subset of machine learning.",
    "Machine learning is a field of artificial intelligence.",
    "NLP is a fascinating area of machine learning.",
    "Natural language processing enables communication between humans and computers.",
    "Word embeddings are a popular way to represent text data."
]

# Schritt 1: Tokenisierung des Korpus
# Tokenisierung ist der Prozess, bei dem Text in einzelne Wörter zerlegt wird
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Schritt 2: Hyperparameter für das Word2Vec-Modell festlegen
embedding_size = 100  # Dimensionen der Wortembeddings
window_size = 5       # Kontextfenstergröße
min_word_count = 1    # Minimale Wortanzahl, um in das Vokabular aufgenommen zu werden
sg = 1                # 1 für Skip-Gram, 0 für Continuous Bag of Words (CBOW)

# Schritt 3: Trainieren des Word2Vec-Modells
model = Word2Vec(sentences=tokenized_corpus,
                 vector_size=embedding_size,
                 window=window_size,
                 min_count=min_word_count,
                 sg=sg)

# Schritt 4: Speichern des Modells
# model.save("word2vec_model.bin")
# print("Das Word2Vec-Modell wurde gespeichert.")

# Schritt 5: Verwendung des Modells
# Ähnlichkeit zwischen Wörtern ermitteln
word = "machine"
if word in model.wv:
    similar_words = model.wv.most_similar(word, topn=5)
    print(f"Die fünf ähnlichsten Wörter zu '{word}':")
    for similar_word, similarity in similar_words:
        print(f"{similar_word}: {similarity:.4f}")
else:
    print(f"Das Wort '{word}' wurde nicht im Vokabular gefunden.")

# Schritt 6: Anzeigen der Vektoren für ein Wort
word_vector = model.wv['deep']
print(f"Vektor für das Wort 'deep':\n{word_vector}")

# Hinweis: Um das Modell in einer realen Anwendung zu verwenden, sollte eine größere und vielfältigere Textsammlung verwendet werden.
# Für ernsthafte Projekte könnte auch die Preprocessing-Schritt (Entfernen von Stoppwörtern, Lemmatisierung etc.) wichtig sein.

# Schritt 7: Aufräumen
del model  # Modell wird gelöscht, um Speicher zu sparen
