import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
from tf_keras.models import Sequential
from tf_keras import models
from tf_keras.layers import Dense, Dropout, BatchNormalization
from tf_keras import layers
from tf_keras.utils import to_categorical
from tf_keras import datasets
from tf_keras.callbacks import TensorBoard
from tf_keras.callbacks import EarlyStopping
from tf_keras.callbacks import ModelCheckpoint
import mlflow
import mlflow.sklearn
import os

mlflow.start_run(run_name="fashion_mnist")

# 1. Fashion-MNIST-Daten laden
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

# Vorverarbeitung der Daten
# Umformen und Normalisieren der Daten
# x_train = x_train.reshape(-1, 28 * 28)  # Umformen zu (60000, 784)
# x_test = x_test.reshape(-1, 28 * 28)    # Umformen zu (10000, 784)
#
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)  # Standardisieren der Trainingsdaten
# x_test = scaler.transform(x_test)        # Standardisieren der Testdaten

x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

# One-Hot-Encoding der Labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


def create_model(n_units):
    model = Sequential()

    # Erste Convolutional-Schicht mit 32 Filtern und 3x3 Kernel
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # Max-Pooling-Schicht zur Reduzierung der Dimensionsanzahl
    model.add(layers.MaxPooling2D((2, 2)))

    # Zweite Convolutional-Schicht mit 64 Filtern
    model.add(layers.Conv2D(64, (3, 3), activation='selu'))
    # Zweite Max-Pooling-Schicht
    model.add(layers.MaxPooling2D((2, 2)))

    # Dritte Convolutional-Schicht mit 64 Filtern
    model.add(layers.Conv2D(64, (3, 3), activation='selu'))

    # Flatten-layer konvertiert die 3D-Ausgabe in eine 1D-Ausgabe
    model.add(layers.Flatten())
    # Vollständig verbundene Schicht
    model.add(layers.Dense(n_units, activation='selu'))
    # Ausgabeschicht mit 10 Neuronen (für 10 Klassen)
    model.add(layers.Dense(10, activation='softmax'))

    # Kompilieren des Modells
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

N_UNITS = 250
mlflow.log_param("n_units", N_UNITS)

model = create_model(N_UNITS)
model.summary()

# Erstellen eines Log-Verzeichnisses für TensorBoard
log_dir = "logs/fit/fashion_mnist/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Verzeichnis für das Speichern der Checkpoints
CHECKPOINT_DIR = './checkpoints/fashion_mnist'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Initialisieren des Checkpoints - speichert nur die besten Modelle
checkpoint = ModelCheckpoint(filepath=os.path.join(CHECKPOINT_DIR, 'cp-{epoch:04d}.weights.h5'),
                             save_weights_only=True,
                             save_best_only=True,
                             verbose=1,
                             monitor='val_accuracy')

BATCH_SIZE = 8
mlflow.log_param("batch_size", BATCH_SIZE)

model.load_weights("checkpoints/fashion_mnist/cp-0011.weights.h5")

model.fit(x_train, y_train,
          epochs=200,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback, early_stopping, checkpoint],
          verbose=1)

import tf_keras
model.save("saved_models/fashion_mnist")
model = tf_keras.models.load_model("saved_models/fashion_mnist")
model.predict(x_test)

model.fit()

loss, acc = model.evaluate(x_test, y_test)
mlflow.log_metric("accuracy", acc)

mlflow.sklearn.log_model(model, "cnn_model")

mlflow.end_run()

# import mlflow
# import pandas as pd
# logged_model = 'runs:/7b9435ee553e4db5b00083a96429a4f6/cnn_model'
#
# # Load model as a PyFuncModel.
# loaded_model = mlflow.pyfunc.load_model(logged_model)
#
# loaded_model.predict(pd.DataFrame(x_test))
#
# x_test.shape

# model = create_model(N_UNITS)
#
# model.load_weights("checkpoints/fashion_mnist/cp-0011.weights.h5")
#
# loss, acc = model.evaluate(x_test, y_test)
#
# model.fit(x_train, y_train,
#           epochs=2,)