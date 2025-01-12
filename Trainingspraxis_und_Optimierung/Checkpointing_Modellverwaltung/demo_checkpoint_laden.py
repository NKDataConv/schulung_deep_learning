"""
Aufgabenstellung: 
In dieser Übung sollen Sie lernen, wie man Checkpoints während des Trainings eines Deep Learning Modells setzt und wie man
einen Trainingslauf mittels dieser Checkpoints fortsetzen kann. Dazu sollen Sie die folgenden Schritte ausführen:

1. Erstellen Sie ein einfaches neuronales Netzwerk mit Keras, um das MNIST-Datenset zu klassifizieren.
2. Implementieren Sie Checkpointing, um das Modell während des Trainings nach jeder Epoche zu speichern.
3. Unterbrechen Sie den Trainingsprozess nach einigen Epochen.
4. Laden Sie das gespeicherte Modell erneut und setzen den Trainingsprozess fort.
5. Beobachten Sie die Auswirkungen des fortgesetzten Trainings auf die Genauigkeit und den Verlust des Modells.
6. Dokumentieren Sie jede wichtige Zeile im Code durch Kommentare, um die Schritte klar zu machen.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import mnist
import os

# Laden des MNIST-Datensets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalisieren der Daten

# Definieren eines sehr einfachen neuronalen Netzwerks
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Um die 2D-Bilder in einen 1D-Array zu konvertieren
    Dense(128, activation='relu'),  # Erste verborgene Schicht mit 128 Neuronen und ReLU Aktivierung
    Dense(10, activation='softmax')  # Ausgangsschicht mit 10 Neuronen für jede Ziffer und Softmax Aktivierung
])

# Kompilieren des Modells mit Optimierer, Verlustfunktion und Metriken
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Verzeichnis für das Speichern der Checkpoints
checkpoint_dir = './checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Initialisieren des Checkpoints - speichert nur die besten Modelle
checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt'),
                             save_weights_only=True,
                             verbose=1,
                             save_best_only=True,
                             monitor='val_accuracy')

# Training des Modells, unterbrechen nach einigen Epochen
print("Starte das erste Training...")
model.fit(x_train, y_train,
          epochs=3,  # Erste Phase: Trainiere für 3 Epochen
          validation_data=(x_test, y_test),
          callbacks=[checkpoint])

# Denken Sie daran, den aktuellen Zustand des Modells zur späteren Fortsetzung zu speichern.

# Erstellen eines neuen Modells derselben Architektur - für den Fall, dass das Skript neu gestartet wird
model_new = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model_new.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Laden der Gewichte vom letzten gespeicherten Checkpoint
latest = tf.train.latest_checkpoint(checkpoint_dir)
model_new.load_weights(latest)

# Fortsetzen des Trainings
print("Fortsetzen des Trainings...")
model_new.fit(x_train, y_train,
              epochs=5,  # Setzt das Training bis zur 8. Epoche fort
              initial_epoch=3,  # Setzt das Training ab der 4. Epoche fort
              validation_data=(x_test, y_test))

# Beachten Sie die Verbesserungen der Genauigkeit durch das Fortsetzen des Trainings
