# Import notwendiger Bibliotheken
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import os
import numpy as np
import pickle
import datetime
import plotly.graph_objects as go

# Definiere grundlegende Parameter
# IMG_WIDTH, IMG_HEIGHT = 224, 224  # Größe der Bilder, die VGG16 erwartet
NUM_CLASSES = 10  # Anzahl der Klassen im benutzerdefinierten Datensatz
BATCH_SIZE = 32  # Anzahl der Bilder pro Batch
EPOCHS = 20  # Anzahl der Trainingsepochen

# Lade das CIFAR-10 Dataset
# falls SSL Fehler -> downloaden und in /data Ordner verschieben
(X_train, y_train), (X_val, y_val) = cifar10.load_data()

# def load_cifar10_data(data_dir):
#     (X_train, y_train) = None, None
#     for batch in range(1, 6):
#         with open(os.path.join(data_dir, f'data_batch_{batch}'), 'rb') as f:
#             batch_data = pickle.load(f, encoding='bytes')
#             if X_train is None:
#                 X_train = batch_data[b'data']
#                 y_train = batch_data[b'labels']
#             else:
#                 X_train = np.vstack((X_train, batch_data[b'data']))
#                 y_train += batch_data[b'labels']
#
#     with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
#         test_data = pickle.load(f, encoding='bytes')
#         X_val = test_data[b'data']
#         y_val = test_data[b'labels']
#
#     X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Reshape to (N, H, W, C)
#     X_val = X_val.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Reshape to (N, H, W, C)
#
#     return (X_train, np.array(y_train)), (X_val, np.array(y_val))
#
# # Lade die Daten
# (X_train, y_train), (X_val, y_val) = load_cifar10_data('daten/cifar-10-batches-py')

# Normalisiere die Pixelwerte
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# Konvertiere die Labels in kategorische Form
y_train = to_categorical(y_train, NUM_CLASSES)
y_val = to_categorical(y_val, NUM_CLASSES)

# Datenvorverarbeitung und Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,  # Rotationsbereich in Grad
    width_shift_range=0.2,  # Horizontale Verschiebung
    height_shift_range=0.2,  # Vertikale Verschiebung
    shear_range=0.2,  # Scherbereich
    zoom_range=0.2,  # Zoom-Bereich
    horizontal_flip=True,  # Horizontal spiegeln
    fill_mode='nearest'  # Füllmodus für leere Pixel
)

val_datagen = ImageDataGenerator()  # Keine Normalisierung mehr nötig

# Erstelle Datengeneratoren für das Training und die Validierung
train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

# Transfer Learning: Lade ein vortrainiertes MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Setze einige der oberen Schichten des Basis-Modells auf trainierbar
for layer in base_model.layers[:-8]:  # Trainiere die letzten 10 Schichten
    layer.trainable = False

# Erstelle das vollständige Modell
model = Sequential()
model.add(base_model)  # Füge das vortrainierte Modell hinzu
model.add(Flatten())  # Flate die Ausgaben des Basis-Modells
model.add(Dense(256, activation='relu'))  # Füge eine dichte Schicht hinzu
model.add(Dense(NUM_CLASSES, activation='softmax'))  # Ausgabeschicht mit Softmax für mehrere Klassen

# Kompiliere das Modell
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callback zur frühzeitigen Beendigung
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Erstellen des TensorBoard-Log-Verzeichnisses
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# TensorBoard-Callback hinzufügen
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Trainiere das Modell
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, tensorboard_callback]
)

# Optional: Speichere das Modell
model.save('fine_tuned_model.keras')
