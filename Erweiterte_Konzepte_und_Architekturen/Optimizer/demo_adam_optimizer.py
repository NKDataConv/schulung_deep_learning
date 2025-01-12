import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import datetime

# Laden des MNIST-Datensatzes
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Vorverarbeitung der Daten
# Umformen und Normalisieren der Daten
x_train = x_train.reshape(-1, 28 * 28)  # Umformen zu (60000, 784)
x_test = x_test.reshape(-1, 28 * 28)    # Umformen zu (10000, 784)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # Standardisieren der Trainingsdaten
x_test = scaler.transform(x_test)        # Standardisieren der Testdaten

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Modell erstellen
def create_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


# Optimizer testen
def train_model(optimizer, learning_rate, log_dir):
    model = create_model()
    optimizer_instance = optimizer(learning_rate=learning_rate)
    model.compile(optimizer=optimizer_instance,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # TensorBoard Callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Modell trainieren
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])


# Lernraten und Optimizer definieren
learning_rates = [0.01, 0.001, 0.0001]
optimizers = {
    "SGD": tf.keras.optimizers.SGD,
    "Adam": tf.keras.optimizers.Adam
}

# Training starten
for optimizer_name, optimizer_class in optimizers.items():
    for lr in learning_rates:
        log_dir = f"logs/fit/{optimizer_name}_lr{lr}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"Training with {optimizer_name}, Learning Rate: {lr}")
        train_model(optimizer_class, lr, log_dir)
