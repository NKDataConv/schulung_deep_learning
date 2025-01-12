# Importieren der benötigten Bibliotheken
import numpy as np
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# Setzen des Zufalls-Seed für Reproduzierbarkeit
np.random.seed(42)

# Daten laden (MNIST-Datensatz)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Vorverarbeitung der Daten
# Umformen und Normalisieren der Daten
x_train = x_train.reshape(-1, 28 * 28)  # Umformen zu (60000, 784)
x_test = x_test.reshape(-1, 28 * 28)    # Umformen zu (10000, 784)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # Standardisieren der Trainingsdaten
x_test = scaler.transform(x_test)        # Standardisieren der Testdaten

# One-Hot-Encoding der Labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Funktion zur Erstellung des MLP-Modells
def create_model(dropout_rate=0.0):
    """
    Erstellen eines MLP-Modells mit optionalem Dropout

    Args:
    dropout_rate (float): Der Dropout-Wert (0.0 bedeutet kein Dropout)

    Returns:
    model: Kompiliertes Keras-Modell
    """
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(28 * 28,)))
    
    # Dropout-Schicht hinzufügen, falls dropout_rate > 0
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(64, activation='relu'))
    
    # Dropout-Schicht hinzufügen
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(10, activation='softmax'))
    
    # Kompilieren des Modells
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# Hyperparameter definieren
epochs = 10
batch_size = 128
dropout_rate = 0.5

# Erstellen der Modelle
model_without_dropout = create_model(dropout_rate=0.0)
model_without_dropout.summary()
model_with_dropout = create_model(dropout_rate=dropout_rate)
model_with_dropout.summary()

# Training des Modells ohne Dropout
history_without_dropout = model_without_dropout.fit(x_train, y_train, 
                                            epochs=epochs, 
                                            batch_size=batch_size, 
                                            validation_split=0.2, 
                                            verbose=1)

# Training des Modells mit Dropout
history_with_dropout = model_with_dropout.fit(x_train, y_train,
                                              epochs=epochs, 
                                              batch_size=batch_size, 
                                              validation_split=0.2, 
                                              verbose=1)

# Funktion zur Darstellung der Trainingshistorie
def plot_history(history1, history2):
    """
    Plotten der Trainingshistorie für Verlust und Genauigkeit
    
    Args:
    history1: Historie des ersten Modells
    history2: Historie des zweiten Modells
    """
    fig = go.Figure()

    # Verlust plotten
    fig.add_trace(go.Scatter(x=list(range(1, len(history1.history['loss']) + 1)), 
                             y=history1.history['loss'], 
                             mode='lines', 
                             name='Loss ohne Dropout'))
    fig.add_trace(go.Scatter(x=list(range(1, len(history2.history['loss']) + 1)), 
                             y=history2.history['loss'], 
                             mode='lines', 
                             name='Loss mit Dropout'))

    fig.update_layout(title='Modell Verlust',
                      xaxis_title='Epoche',
                      yaxis_title='Verlust')

    fig.show()

    # Genauigkeit plotten
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(history1.history['accuracy']) + 1)), 
                             y=history1.history['accuracy'], 
                             mode='lines', 
                             name='Genauigkeit ohne Dropout'))
    fig.add_trace(go.Scatter(x=list(range(1, len(history2.history['accuracy']) + 1)), 
                             y=history2.history['accuracy'], 
                             mode='lines', 
                             name='Genauigkeit mit Dropout'))

    fig.update_layout(title='Modellgenauigkeit',
                      xaxis_title='Epoche',
                      yaxis_title='Genauigkeit')

    fig.show()

# Plotten der Ergebnisse
plot_history(history_without_dropout, history_with_dropout)

# Testen der Modelle mit dem Testdatensatz
test_loss_no_dropout, test_acc_no_dropout = model_without_dropout.evaluate(x_test, y_test, verbose=0)
test_loss_with_dropout, test_acc_with_dropout = model_with_dropout.evaluate(x_test, y_test, verbose=0)

# Ausgeben der Testergebnisse
print(f"Testgenauigkeit ohne Dropout: {test_acc_no_dropout:.4f}")
print(f"Testgenauigkeit mit Dropout: {test_acc_with_dropout:.4f}")
