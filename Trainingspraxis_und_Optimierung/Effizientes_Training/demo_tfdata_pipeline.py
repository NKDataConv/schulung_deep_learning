import tensorflow as tf
import numpy as np

# Beispiel-Daten für die Textanalyse
texts = [
    "Deep Learning ist ein Teilbereich von Machine Learning.",
    "Text Mining und Verarbeitung sind wichtig für NLP.",
    "Transformer-Modelle haben die Art und Weise revolutioniert, wie wir Textverarbeitung angehen.",
    "Effizientes Training ist entscheidend für tiefes Lernen.",
]

# Labels für die Daten (z.B. 0 = negativ, 1 = positiv)
labels = np.array([1, 1, 1, 0])

# Hyperparameter
BATCH_SIZE = 2
AUTOTUNE = tf.data.AUTOTUNE

# Funktion zum Vorverarbeiten der Texte
def preprocess_text(text):
    # Konvertiere in Kleinbuchstaben
    text = tf.strings.lower(text)
    # Tokenisierung (einfaches Beispiel, bessere Tokenisierung ist möglich)
    text = tf.strings.regex_replace(text, '[^a-zA-Z0-9 ]', '')
    return text

# Erstelle TensorFlow Dataset aus den Texten und Labels
def create_dataset(texts, labels):
    # Erstelle ein tf.data Dataset
    data = tf.data.Dataset.from_tensor_slices((texts, labels))
    # Wende die Vorverarbeitung an
    data = data.map(lambda text, label: (preprocess_text(text), label), num_parallel_calls=AUTOTUNE)
    # Mische die Daten und teile sie in Batches auf
    data = data.shuffle(buffer_size=len(texts)).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return data

# Bereite die Trainingsdaten vor
train_dataset = create_dataset(texts, labels)

# Beispiel für ein einfaches neuronales Netzwerk
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,), dtype=tf.string),
    tf.keras.layers.TextVectorization(),
    tf.keras.layers.Embedding(input_dim=1000, output_dim=64),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Kompiliere das Modell
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Trainiere das Modell (auf unserem kleinen Datensatz)
model.fit(train_dataset, epochs=5)

# Demonstration der Vorhersage
def predict(text):
    # Vorverarbeitung des Eingabetextes
    text = preprocess_text(tf.constant([text]))
    prediction = model.predict([text])
    return prediction

# Beispielvorhersage
input_text = "Deep Learning revolutioniert das Lernen."
prediction = predict(input_text)

print(f"Vorhersage für: '{input_text}' ist {prediction[0][0]:.4f} (Wahrscheinlichkeit für positiv)")

# **Erklärung des Skripts:**
# 1. **Imports:** TensorFlow und NumPy werden importiert, um das Modell zu erstellen und mit Daten zu arbeiten.
# 2. **Daten:** Beispieltexte und ihre zugehörigen Labels werden definiert.
# 3. **Hyperparameter:** Batch-Größe und AUTOTUNE für eine optimierte Datensatzbearbeitung sind festgelegt.
# 4. **Textvorverarbeitung:** Eine Funktion `preprocess_text` fördert die Standardisierung von Texten.
# 5. **Datenpipeline:** `create_dataset` erstellt eine Datensatz-Pipeline unter Verwendung von `tf.data`, die Vorverarbeitung, Mischen und Batching enthält.
# 6. **Modellerstellung:** Ein einfaches neuronales Netzwerk wird definiert, das Textdaten verarbeitet.
# 7. **Trainingsprozess:** Das Modell wird kompiliert und mit den Trainingsdaten trainiert.
# 8. **Vorhersagefunktion:** Eine Funktion ermöglicht die Vorhersage für neue Texte.
# 9. **Beispielvorhersage:** Eine Beispielvorhersage wird durchgeführt und das Ergebnis wird ausgegeben.
