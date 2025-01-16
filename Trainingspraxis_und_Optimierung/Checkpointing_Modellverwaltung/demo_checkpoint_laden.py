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
CHECKPOINT_DIR = './checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Initialisieren des Checkpoints - speichert nur die besten Modelle
checkpoint = ModelCheckpoint(filepath=os.path.join(CHECKPOINT_DIR, 'cp-{epoch:04d}.weights.h5'),
                             save_weights_only=True,
                             verbose=1,
                             save_best_only=True,
                             monitor='val_accuracy')

# Training des Modells, unterbrechen nach einigen Epochen
print("Starte das erste Training...")
model.fit(x_train, y_train,
          epochs=3,  # Erste Phase: Trainiere für 3 Epochen
          validation_data=(x_test, y_test),
          verbose=1,
          callbacks=[checkpoint])

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
latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if latest:
    try:
        model_new.load_weights(latest)
        print(f"Gewichte von {latest} erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden der Gewichte: {e}")

# Fortsetzen des Trainings
print("Fortsetzen des Trainings...")
model_new.fit(x_train, y_train,
              epochs=5,  # Setzt das Training bis zur 5. Epoche fort
              initial_epoch=3,  # Setzt das Training ab der 4. Epoche fort
              verbose=1,
              validation_data=(x_test, y_test))

# Ausgabe:
pred = model_new.predict(x_test[:1])  # Beispielvorhersage
print("Vorhersage:", pred.argmax())  # Ausgabe der Vorhersage
