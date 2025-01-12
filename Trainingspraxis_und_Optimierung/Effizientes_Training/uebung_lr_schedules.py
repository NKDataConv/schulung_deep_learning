"""
Aufgabenstellung:
In dieser Übung sollen die Schulungsteilnehmer die Effektivität verschiedener Learning Rate Schedules 
beim Training eines einfachen neuronalen Netzwerks auf einem Textklassifizierungsproblem bewerten. 
Ziel ist es, die Auswirkungen verschiedener Strategien zur Anpassung der Lernrate während des Trainings 
auf die Modellgenauigkeit zu analysieren. Die Teilnehmer werden gebeten, drei verschiedene Learning Rate
Schedules zu implementieren und zu vergleichen: Konstant, Step Decay und Exponentiell. Die Ergebnisse
werden dann in Form von Genauigkeitskurven über die Epochen hinweg grafisch dargestellt.

Schritte:
1. Laden Sie ein Textklassifizierungs-Dataset, z.B. das Reuters-Dataset.
2. Vorverarbeiten Sie die Textdaten für die Eingabe in ein neuronales Netzwerk.
3. Implementieren Sie ein einfaches neuronales Netzwerkmodell.
4. Trainieren Sie das Modell mit einem konstanten Learning Rate Schedule und zeichnen Sie die Trainingskurve auf.
5. Implementieren Sie einen Step Decay Learning Rate Schedule und trainieren Sie das Modell erneut.
6. Implementieren Sie einen Exponentiellen Learning Rate Schedule und trainieren Sie das Modell noch einmal.
7. Vergleichen Sie die Ergebnisse der verschiedenen Schedules grafisch.

Verwenden Sie Python und relevante Bibliotheken wie TensorFlow/Keras, Pandas und Matplotlib.
"""

import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Laden des Reuters-Datasets
# Zur Vereinfachung verwenden wir nur die Daten und Labels
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

# Datenvorverarbeitung: Sequenzen in Vektoren umwandeln
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

# Labels in kategorische Form umwandeln
num_classes = max(y_train) + 1
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Modell erstellen
def create_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

# Funktion zum Plotten der Trainingsgeschichte
def plot_history(history, schedule_name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Modellgenauigkeit mit {schedule_name} Learning Rate Schedule')
    plt.ylabel('Genauigkeit')
    plt.xlabel('Epoche')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

# 1. Konstant
model_const = create_model()
model_const.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

# Trainiert das Modell mit konstantem Lernraten-Schedule
history_const = model_const.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), verbose=2)
plot_history(history_const, "Konstantem")

# 2. Step Decay
def step_decay_schedule(epoch, lr):
    """Learnrate wird bei jeder 5. Epoche halbiert"""
    drop = 0.5
    epochs_drop = 5.0
    return lr * (drop ** (epoch // epochs_drop))

model_step = create_model()
model_step.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

lr_scheduler_step = tf.keras.callbacks.LearningRateScheduler(step_decay_schedule, verbose=1)
history_step = model_step.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), callbacks=[lr_scheduler_step], verbose=2)
plot_history(history_step, "Step Decay")

# 3. Exponentiell
def exponential_decay_schedule(epoch, lr):
    """Learnrate wird exponentiell mit einem festgelegten konstanten Faktor verringert"""
    k = 0.1
    return lr * tf.math.exp(-k * epoch)

model_expo = create_model()
model_expo.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

lr_scheduler_expo = tf.keras.callbacks.LearningRateScheduler(exponential_decay_schedule, verbose=1)
history_expo = model_expo.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), callbacks=[lr_scheduler_expo], verbose=2)
plot_history(history_expo, "Exponentialer Decay")

# Ergebnisse können nun grafisch verglichen werden, um die Auswirkungen verschiedener Lernratenstrategien zu evaluieren
