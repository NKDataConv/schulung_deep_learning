# Übung: Sentiment Analysis-Modell mit eigenen Datensätzen trainieren
#
# Aufgabenstellung:
#
# Ziel dieser Übung ist es, ein einfaches Sentiment-Analyse-Modell zu entwickeln, das in der Lage ist, 
# die Stimmung von Texten als positiv oder negativ zu klassifizieren. Dazu sollten die Schulungsteilnehmer:
# 1. Einen eigenen Textdatensatz vorbereiten, der aus mindestens 1000 Dokumenten besteht. Jedes Dokument sollte
#    einem positiven oder negativen Label zugeordnet sein. Beispiel: Filmrezensionen, Tweets, Produktbewertungen.
# 2. Den Textdatensatz preprocessen, um ihn für das Modelltraining vorzubereiten (Tokenisierung, Bereinigung usw.).
# 3. Ein Deep Learning-Modell (z.B. LSTM, oder ein einfaches Dense-Netzwerk) zur Textklassifizierung implementieren.
# 4. Das Modell mit dem präparierten Datensatz trainieren, validieren und dessen Leistung mittels Metriken wie 
#    Accuracy, Precision und Recall evaluieren.
# 5. Die trainierten Modellparameter speichern, um später Vorhersagen auf neuen Textdaten durchführen zu können.
# 6. Eine kurze Auswertung und Interpretation der Modellleistung durchführen.
#
# Hinweis: Dieses Beispiel implementiert alle erforderlichen Schritte von der Datenvorbereitung bis zur 
# Modellspeicherung und enthält viele Kommentare für ein einfaches Verständnis.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
import os

# Step 1: Datensatz vorbereiten
# Hier ist ein generischer Platzhalter für einen CSV-Datensatz mit zwei Spalten: 'Text' und 'Label'.
# Die Teilnehmer sollten dies mit ihrem eigenen Datensatz ersetzen.
data_path = "path_to_your_dataset.csv"  # Achtung: Muss durch den realen Pfad ersetzt werden.
data = pd.read_csv(data_path)

# Erzwingen der Balance im Datensatz, wenn dies notwendig ist (optional, abhängig vom Datensatz)
# min_size = min(data["Label"].value_counts())
# balanced_data = pd.concat([
#     data[data["Label"] == 'positive'].sample(min_size, random_state=42),
#     data[data["Label"] == 'negative'].sample(min_size, random_state=42)
# ])

# Step 2: Textvorverarbeitung
# Konvertieren der Labels in numerische Werte
label_encoder = LabelEncoder()
data["Label"] = label_encoder.fit_transform(data["Label"])

# Textdaten tokenisieren und in Sequenzen umwandeln
max_words = 10000  # Maximale Anzahl der Wörter im Vokabular
max_len = 100  # Maximale Länge der Sequenzen

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data["Text"])
sequences = tokenizer.texts_to_sequences(data["Text"])
X = pad_sequences(sequences, maxlen=max_len)
y = data["Label"]

# Aufteilen der Daten in Training und Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Modellentwicklung
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Nutzung einer Sigmoid-Aktivierung für Binärklassifikation

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Step 4: Modelltraining
checkpoint_path = "sentiment_analysis.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

# Laden des besten gespeicherten Modells
model.load_weights(checkpoint_path)

# Step 5: Evaluierung des Modells
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

# Step 6: Auswertung und Interpretation
# - Hohe Accuracy-Werte weisen auf ein gutes Modell hin. Bei hohem Klassenungleichgewicht sind Precision und Recall wichtig.
# - Überprüfe die Confusion Matrix (nicht implementiert) für tiefere Einblicke in die Klassifikationsleistung.
# - Spare den Tokenizer, wenn Du das Modell auf neuen Daten verwenden möchtest.
# Optional: Tokenizer speichern
tokenizer_path = "tokenizer.pkl"
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Modell und Tokenizer gespeichert, die Durchführung der Sentiment Analyse auf neuen Datensätzen ist möglich.")
