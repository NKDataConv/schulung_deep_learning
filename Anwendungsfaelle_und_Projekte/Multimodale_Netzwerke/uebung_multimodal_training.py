# Übung: Training eines multimodalen Modells auf einem eigenen Bilddatensatz
# Ziel dieser Übung ist es, ein einfaches multimodales Modell zu erstellen, das in der Lage ist, zu einem gegebenen Bild einen passenden Untertitel zu generieren.
# Hierbei werden Bild- und Textdaten gemeinsam verarbeitet, um ein tieferes Verständnis von multimodalen Netzwerken zu erlangen.
# Die Teilnehmer sollen:
# 1. Einen Bilddatensatz laden (die Bilder können selbst ausgewählt oder ein verfügbarer Datensatz genutzt werden, z.B. COCO oder Flickr8k).
# 2. Einen einfachen Text-Datensatz erstellen, der zu jedem Bild eine oder mehrere Beschreibungen (Untertitel) enthält.
# 3. Ein Preprocessing für die Bild- und Textdaten durchführen.
# 4. Ein einfaches multimodales Modell mithilfe von Keras oder PyTorch implementieren.
# 5. Das Modell trainieren und evaluieren.
# 6. Ergebnisse analysieren und verbessern.

# Hinweis: Dies ist eine vereinfachte Übung, um das Grundkonzept zu verstehen. Es können bestehende Modelle und Bibliotheken wie Hugging Face Transformers oder TensorFlow genutzt werden, um den Umfang der Implementierung zu vereinfachen.

# Im Folgenden ist ein vereinfachtes Beispielskript beschrieben, das diese Aufgaben mit Keras löst.

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout, add
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Diese Funktion lädt ein Bild, konvertiert es in ein Array und wandelt es in die geeignete Form um.
def load_image(filename, target_size=(299, 299)):
    image = load_img(filename, target_size=target_size)
    image = img_to_array(image)
    # Normalisiere die Bilddaten, damit sie in den Bereich [0,1] fallen
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Beispiel Text-Datensatz (Zu jedem Bild gibt es einen einfachen Untertitel. In der Praxis sollten hier natürlich mehr Beispieluntertitel stehen.)
captions = {
    'image_1.jpg': 'A dog playing with a ball in the yard.',
    'image_2.jpg': 'A cat sitting on a window sill watching birds.',
    # Hier könnten viele weitere Bilddateien und Beschreibungen stehen
}

# Laden und Vorverarbeiten der Daten
image_dir = '/path/to/image/directory/'  # Verzeichnis, in dem sich die Bilder befinden
image_model = Xception(include_top=False, pooling='avg')

# Erstellen eines Bild-Feature-Vektors für jedes Bild
image_features = {}
for image_file, caption in captions.items():
    image = load_image(image_dir + image_file)
    image_features[image_file] = image_model.predict(image)

# Tokenizer für die Textdaten einrichten
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions.values())

# Vorbereitung der Trainingsdaten
max_length = max(len(caption.split()) for caption in captions.values())

# Funktion zur Ausgabe der Eingabesequenzen für das Modell
def create_sequences(tokenizer, max_length, image_features, captions):
    """ 
    Erstellt Trainingssequenzen für die Texte aus den Bild-, Textdaten- und Wortindexdaten.
    """
    X1, X2, y = [], [], []
    vocab_size = len(tokenizer.word_index) + 1

    for image_id, caption in captions.items():
        seq = tokenizer.texts_to_sequences([caption])[0]

        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]

            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]

            X1.append(image_features[image_id][0])
            X2.append(in_seq)
            y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)

X1, X2, y = create_sequences(tokenizer, max_length, image_features, captions)

# Trainings- und Validierungssets erstellen
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

# Modell Architektur
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 256
units = 256

# Bildmodell
image_input = Input(shape=(2048,))
image_dropout = Dropout(0.5)(image_input)
image_dense = Dense(units, activation='relu')(image_dropout)

# Textmodell
text_input = Input(shape=(max_length,))
text_embedding = Embedding(vocab_size, embedding_dim)(text_input)
text_lstm = LSTM(units)(text_embedding)

# Zusammenführen der Modelle
decoder = add([image_dense, text_lstm])
decoder = Dense(units, activation='relu')(decoder)
outputs = Dense(vocab_size, activation='softmax')(decoder)

# Kompletter Modellaufbau
model = Model(inputs=[image_input, text_input], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modelltraining
model.fit([X1_train, X2_train], y_train, epochs=20, batch_size=64, validation_data=([X1_test, X2_test], y_test))

# Auswertung der Ergebnisse
# Die Ergebnisse können z.B. ausgewertet werden, indem Vorhersagen für Testbilder gemacht und manuell mit den bekannten Bilduntertiteln verglichen werden.
