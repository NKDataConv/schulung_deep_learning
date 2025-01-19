# Übungsaufgabe: Verschiedene Aktivierungsfunktionen in einem MLP testen

# Aufgabenstellung:
# In dieser Übung werden wir ein einfaches MLP für ein
# Klassifikationsproblem verwenden und den Einfluss verschiedener Aktivierungs-
# funktionen auf die Modellleistung untersuchen. Ihre Aufgaben sind:
# 1. Verwenden Sie das MNIST-Dataset, das handgeschriebene Ziffern enthält.
# 2. Implementieren Sie ein MLP mit einer einzigen versteckten Schicht.
# 3. Testen Sie drei verschiedene Aktivierungsfunktionen: ReLU, Sigmoid und Tanh.
# 4. Trainieren Sie das Modell mit jedem der Aktivierungsfunktionen.
# 5. Evaluieren und vergleichen Sie die Leistung der verschiedenen Aktivierungs-
#    funktionen anhand ihrer Testgenauigkeit.
# 6. Erstellen Sie grafische Darstellungen der Train- und Testverluste für
#    jede Aktivierungsfunktion.

from tf_keras.datasets import mnist
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Flatten
from tf_keras.utils import to_categorical
from tf_keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Laden Sie den MNIST-Datensatz (enthält handgeschriebene Ziffern)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Vorverarbeitung der Daten: Normalisieren der Pixelwerte und One-Hot-Encoding der Labels
# Umformen der Daten für den StandardScaler
x_train = x_train.reshape(-1, 28 * 28)  # Umformen zu (60000, 784)
x_test = x_test.reshape(-1, 28 * 28)    # Umformen zu (10000, 784)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # Standardisieren der Trainingsdaten
x_test = scaler.transform(x_test)        # Standardisieren der Testdaten

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def create_model(activation_function):
    """
    Creates and returns a MLP model with specified activation function
    
    Args:
        activation_function (str): Activation function to use in hidden layer
    """
    model = Sequential()
    model.add(Dense(128, activation=activation_function, input_shape=(28 * 28,)))  # Hidden layer
    model.add(Dense(10, activation='softmax'))  # Output layer for 10 classes
    
    model.compile(optimizer=Adam(), 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    
    return model

# Funktion zum Erstellen und Trainieren eines MLP-Modells mit einer gegebenen Aktivierungsfunktion
def create_and_train_mlp(activation_function):
    # Modellarchitektur: Einfaches MLP mit einer versteckten Schicht
    model = create_model(activation_function)

    # Modelltraining mit Trainingsdaten
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
    
    # Testgenauigkeit
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Aktivierungsfunktion: {activation_function} - Testgenauigkeit: {test_accuracy:.4f}")

    # Rückgabewert: Historie der Verluste und Genauigkeitswerte
    return history

# Definition der verschiedenen Aktivierungsfunktionen, die getestet werden
activation_functions = ['relu', 'sigmoid', 'tanh', 'elu', 'selu']

# Initialisierung eines Dictionaries zur Aufnahme der Trainingsergebnisse
history_dict = {}

# Schleife über die Aktivierungsfunktionen, um Modelle zu erstellen, zu trainieren und Ergebnisse zu speichern
for activation in activation_functions:
    print(f"Training mit Aktivierungsfunktion: {activation}")
    history = create_and_train_mlp(activation)
    history_dict[activation] = history

# Plotly-Darstellung für Validierungsverlust
fig_loss = go.Figure()
for activation in activation_functions:
    fig_loss.add_trace(go.Scatter(x=list(range(1, len(history_dict[activation].history['val_loss']) + 1)), 
                            y=history_dict[activation].history['val_loss'], 
                            mode='lines', 
                            name=f'val_loss-{activation}'))

fig_loss.update_layout(title='Validierungsverlust für verschiedene Aktivierungsfunktionen',
                        xaxis_title='Epoche',
                        yaxis_title='Validierungsverlust')
fig_loss.show()

# Plotly-Darstellung für Validierungsgenauigkeit
fig_accuracy = go.Figure()
for activation in activation_functions:
    fig_accuracy.add_trace(go.Scatter(x=list(range(1, len(history_dict[activation].history['val_accuracy']) + 1)), 
                                y=history_dict[activation].history['val_accuracy'], 
                                mode='lines', 
                                name=f'val_accuracy-{activation}'))

fig_accuracy.update_layout(title='Validierungsgenauigkeit für verschiedene Aktivierungsfunktionen',
                            xaxis_title='Epoche',
                            yaxis_title='Validierungsgenauigkeit')
fig_accuracy.show()
