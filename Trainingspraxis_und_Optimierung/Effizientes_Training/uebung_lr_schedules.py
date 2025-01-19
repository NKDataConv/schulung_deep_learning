import tensorflow as tf
from tf_keras.datasets import mnist
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from tf_keras.optimizers import SGD
import numpy as np
import plotly.graph_objects as go

EPOCHS = 15
BATCH_SIZE = 32
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

# Laden des MNIST-Datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Datenvorverarbeitung: Normalisierung und Umformung
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Labels in kategorische Form umwandeln
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# CNN-Modell erstellen
def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

# Funktion zum Plotten der Trainingsgeschichte mit Plotly
def plot_history(history, schedule_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['accuracy'], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(y=history.history['val_accuracy'], mode='lines', name='Test'))
    
    fig.update_layout(
        title=f'Modellgenauigkeit mit {schedule_name} Learning Rate Schedule',
        xaxis_title='Epoche',
        yaxis_title='Genauigkeit',
        legend_title='Daten',
        template='plotly'
    )
    
    fig.show()

# 1. Konstant
model_const = create_cnn_model()
model_const.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

# Trainiert das Modell mit konstantem Lernraten-Schedule
history_const = model_const.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test), verbose=2)
plot_history(history_const, "Konstant")

# 2. Step Decay
def step_decay_schedule(epoch, lr):
    """Learnrate wird bei jeder 5. Epoche halbiert"""
    drop = 0.5
    epochs_drop = 5.0
    return lr * (drop ** (epoch // epochs_drop))

model_step = create_cnn_model()
model_step.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

lr_scheduler_step = tf.keras.callbacks.LearningRateScheduler(step_decay_schedule, verbose=1)
history_step = model_step.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=[lr_scheduler_step], verbose=2)
plot_history(history_step, "Step Decay")

# 3. Exponentiell
def exponential_decay_schedule(epoch, lr):
    """Learnrate wird exponentiell mit einem festgelegten konstanten Faktor verringert"""
    k = 0.1
    return lr * np.exp(-k * epoch)

model_expo = create_cnn_model()
model_expo.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

lr_scheduler_expo = tf.keras.callbacks.LearningRateScheduler(exponential_decay_schedule, verbose=1)
history_expo = model_expo.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=[lr_scheduler_expo], verbose=2)
plot_history(history_expo, "Exponentialer Decay")
