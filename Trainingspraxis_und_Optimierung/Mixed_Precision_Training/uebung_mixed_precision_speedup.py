"""
Übung: Mixed Precision Training aktivieren und Speedup messen

Aufgabenstellung:
1. Implementieren Sie ein tiefes neuronales Netzwerk zur Bildklassifikation unter Verwendung des MNIST-Datensatzes.
2. Aktivieren Sie das Mixed Precision Training.
3. Messen Sie die Trainingszeit ohne Mixed Precision Training und mit Mixed Precision Training.
4. Vergleichen Sie die Geschwindigkeit und dokumentieren Sie die Ergebnisse.

Vorgehen:
- Laden und Vorverarbeiten des MNIST-Datensatzes.
- Erstellen Sie ein einfaches neuronales Netzwerk mit mehreren Schichten.
- Trainieren Sie das Modell zunächst ohne Mixed Precision und messen Sie die Zeit.
- Aktivieren Sie das Mixed Precision Training und messen Sie erneut die Zeit.
- Vergleichen Sie die Zeitunterschiede zwischen beiden Methoden.
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import time

# Datensatz laden und vorverarbeiten
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Einfaches Modell erstellen
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# Trainingsfunktion
def train_model(model, mixed_precision=False):
    if mixed_precision:
        # Mixed Precision aktivieren - Verwenden von 'mixed_float16' Policy
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Model kompilieren
    model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Model trainieren
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Zeit messen für Training ohne Mixed Precision
start_time = time.time()
model_standard = create_model()
train_model(model_standard, mixed_precision=False)
standard_time = time.time() - start_time
print(f"Trainingszeit ohne Mixed Precision: {standard_time:.2f} Sekunden")

# Zeit messen für Training mit Mixed Precision
start_time = time.time()
model_mixed = create_model()
train_model(model_mixed, mixed_precision=True)
mixed_time = time.time() - start_time
print(f"Trainingszeit mit Mixed Precision: {mixed_time:.2f} Sekunden")

# Vergleich der Ergebnisse
print(f"Geschwindigkeitszuwachs: {standard_time / mixed_time:.2f}x")
