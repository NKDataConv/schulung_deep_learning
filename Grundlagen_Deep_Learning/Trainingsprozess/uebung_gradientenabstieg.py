# Übung: Einfluss von Lernrate und Epochenanzahl experimentell untersuchen

# Aufgabenstellung:
# In dieser Übung sollen Sie den Einfluss der Lernrate (learning rate) und der Epochenanzahl (epochs)
# auf das Trainingsverhalten eines einfachen Neuronalen Netzes untersuchen. Verwenden Sie dazu das bekannte
# Iris-Datenset, um ein Neuronales Netz zu trainieren. Variieren Sie die Lernrate und die Anzahl der Epochen
# und beobachten Sie, wie sich die Änderungen auf die Genauigkeit des Modells auswirken. Führen Sie 
# Experimente durch, indem Sie verschiedene Werte für die Lernrate und die Anzahl der Epochen verwenden, 
# und dokumentieren Sie die Auswirkungen dieser Variablen auf die Trainings- und Validierungsgenauigkeit. 

# Bonus: Entferne die Skalierung der Eingabedaten und beobachte, wie sich die Genauigkeit des Modells ändert.
# Bonus: Ändere die Anzahl der Neuronen im Neuronalen Netz und beobachte, wie sich die Genauigkeit ändert.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tf_keras.models import Sequential
from tf_keras.layers import Dense
from tf_keras.utils import to_categorical
from tf_keras import optimizers
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Laden des Iris-Datensets
iris = load_iris()
X = iris.data
y = iris.target

# Normalisieren der Eingabedaten
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Aufteilen von Training- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Konvertieren der Zielvariablen in kategorische Daten
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Funktion zum Trainieren und Validieren des neuronalen Netzwerks
def train_nn_with_hyperparams(X_train, y_train, X_test, y_test, epochs, learning_rate):
    # Erstellen des neuronalen Netzwerks
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))  # Eingabeschicht
    model.add(Dense(10, activation='relu'))  # Verborgene Schicht
    model.add(Dense(y_train.shape[1], activation='softmax'))  # Ausgabeschicht

    # Kompilieren des Modells
    optimizer = optimizers.SGD(learning_rate=learning_rate)  # Lernrate
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Modellzusammenfassung anzeigen
    model.summary()

    # Training des Modells
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_data=(X_test, y_test))

    return history.history['accuracy']


# Starten Sie mehrere Experimente mit unterschiedlichen Parametern
learning_rates = [0.1, 0.01, 0.001]  # Learning Rate
epochs_list = [5, 10, 15]  # Anzahl der Epochen

# Ergebnisse speichern
results = {}

# Experimente mit verschiedenen Kombinationen von Parameterwerten durchführen
for learning_rate in learning_rates:
    for epochs in epochs_list:
        print(f"\nTraining with learning_rate={learning_rate} and epochs={epochs}:")
        accuracies = train_nn_with_hyperparams(X_train, y_train_categorical, X_test, y_test_categorical, epochs, learning_rate)
        results[(learning_rate, epochs)] = accuracies

# Visualisieren der Ergebnisse mit Plotly
fig = go.Figure()

for (learning_rate, epochs), accuracies in results.items():
    fig.add_trace(go.Scatter(
        x=list(range(1, epochs + 1)),
        y=accuracies,
        mode='lines+markers',
        name=f'learning_rate={learning_rate}, epochs={epochs}'
    ))

fig.update_layout(
    title='Accuracy over Epochs for different Learning Rates',
    xaxis_title='Epoch',
    yaxis_title='Accuracy',
    legend_title='Parameters',
    template='plotly'
)

fig.show()
