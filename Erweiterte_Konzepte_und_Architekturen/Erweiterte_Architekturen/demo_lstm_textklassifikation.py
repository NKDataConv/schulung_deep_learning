# Importieren der benötigten Bibliotheken
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense, Embedding
from tf_keras.callbacks import TensorBoard
import datetime


# Setzen von Zufallswerten für die Reproduzierbarkeit
np.random.seed(42)

# ------------------------------
# 1. Daten vorbereiten
# ------------------------------

# Beispieldaten: Textbeispiele und zugehörige Labels
data = {
    'text': [
        'Ich liebe Programmierung',
        'Deep Learning ist faszinierend',
        'Python ist eine großartige Programmiersprache',
        'Künstliche Intelligenz wird die Welt verändern',
        'Das Wetter ist heute sonnig',
        'Ich mag Sport und Bewegung',
        'Politik ist ein wichtiges Thema',
    ],
    'label': [
        'positiv',
        'positiv',
        'positiv',
        'positiv',
        'neutral',
        'neutral',
        'neutral',
    ]
}

# Erstellen eines DataFrames
df = pd.DataFrame(data)

# Aufteilen des Datensatzes in Features (X) und Labels (y)
X = df['text']
y = df['label']

# ------------------------------
# 2. Datenaufteilung
# ------------------------------

# Train-Test-Split durchführen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# 3. Textvorverarbeitung
# ------------------------------

# Label-Encoding für die Zielvariable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Tokenizer erstellen
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(X_train)

# Texte in Sequenzen umwandeln
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding der Sequenzen
max_length = 10  # Maximale Länge der Sequenzen
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# ------------------------------
# 4. LSTM Modell erstellen
# ------------------------------

def create_model():
    """
    Creates and returns an LSTM model for text classification
    
    Returns:
        model: Compiled LSTM model ready for training
    """
    model = Sequential()
    # Embedding-Layer hinzufügen
    model.add(Embedding(input_dim=1000, output_dim=64, input_length=max_length))
    # LSTM-Layer hinzufügen
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    # Dense-Layer für die Klassifikation hinzufügen
    model.add(Dense(3, activation='softmax'))  # 3 Klassen für positiv, neutral, negativ
    # Modell kompilieren
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model    

# Modell initialisieren
model = create_model()

# ------------------------------
# 5. Modelltraining
# ------------------------------
# Erstellen des TensorBoard-Log-Verzeichnisses
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# TensorBoard-Callback hinzufügen
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Modell trainieren
history = model.fit(X_train_padded, y_train_encoded, 
                    epochs=10, 
                    batch_size=2, 
                    validation_data=(X_test_padded, y_test_encoded), 
                    callbacks=[tensorboard_callback])

# ------------------------------
# 6. Modellbewertung
# ------------------------------

# Modellbewertung auf Testdaten
loss, accuracy = model.evaluate(X_test_padded, y_test_encoded)

print(f'\nTestverlust: {loss:.4f}')
print(f'Testgenauigkeit: {accuracy:.4f}')

# ------------------------------
# 7. Vorhersagen treffen
# ------------------------------

# Beispiele für Vorhersagen
texts_to_predict = [
    'Heute ist ein schöner Tag',
    'Ich hasse es, wenn Wintertage so kurz sind.',
]

# Vorverarbeitung der vorherzusagenden Texte
texts_seq = tokenizer.texts_to_sequences(texts_to_predict)
texts_padded = pad_sequences(texts_seq, maxlen=max_length, padding='post')

# Vorhersagen durchführen
predictions = model.predict(texts_padded)

# Ergebnisse anzeigen
for i, text in enumerate(texts_to_predict):
    print(f'Text: "{text}" -> Vorhersage: "{label_encoder.inverse_transform([np.argmax(predictions[i])])[0]}"')
