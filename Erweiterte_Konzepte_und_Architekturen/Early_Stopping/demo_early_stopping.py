# Importiere benötigte Bibliotheken
import numpy as np
import matplotlib.pyplot as plt
from tf_keras.models import Sequential
from tf_keras.layers import Dense
from tf_keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from tf_keras.callbacks import TensorBoard
import datetime
import plotly.graph_objects as go


# Erstelle einen Datensatz für die Demo
# Hier verwenden wir den 'make_moons'-Datensatz, der geeignet für Klassifikationsaufgaben ist
X, y = make_moons(n_samples=10000, noise=0.1, random_state=42)

# Teile den Datensatz in Trainings- und Testdaten auf
# 80% für das Training und 20% für das Testen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Erstelle ein einfaches neuronales Netzwerkmodell
def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=2, activation='relu'))  # Erster Hidden Layer mit 10 Neuronen
    model.add(Dense(1, activation='sigmoid'))  # Ausgabe Layer für binäre Klassifikation

    # Kompiliere das Modell
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Definiere den Early Stopping Callback
# Wir wollen das Training beenden, wenn sich die Validierungsgenauigkeit nicht mehr verbessert
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Erstellen des TensorBoard-Log-Verzeichnisses
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# TensorBoard-Callback hinzufügen
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Trainiere das Modell mit den Trainingsdaten
# Wir verwenden 20% der Trainingsdaten für die Validierung
model = create_model()
history = model.fit(X_train, y_train, 
                    validation_split=0.2, 
                    epochs=200,  # Maximale Anzahl der Epochen
                    batch_size=10, 
                    callbacks=[early_stopping, tensorboard_callback])  # Füge den Early Stopping Callback hinzu

# Auswertung des Modells auf den Testdaten
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Testverlust: {test_loss:.4f}, Testgenauigkeit: {test_accuracy:.4f}')

# Plotten der Trainingshistorie
# Zeige die Verlust- und Genauigkeitskurven über die Epochen
fig = go.Figure()

# Verlust plotten
fig.add_trace(go.Scatter(x=list(range(len(history.history['loss']))), 
                         y=history.history['loss'], 
                         mode='lines', 
                         name='Trainingsverlust'))
fig.add_trace(go.Scatter(x=list(range(len(history.history['val_loss']))), 
                         y=history.history['val_loss'], 
                         mode='lines', 
                         name='Validierungsverlust'))

fig.update_layout(title='Verlust über Epochen',
                  xaxis_title='Epoche',
                  yaxis_title='Verlust')

# Genauigkeit plotten
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=list(range(len(history.history['accuracy']))), 
                          y=history.history['accuracy'], 
                          mode='lines', 
                          name='Trainingsgenauigkeit'))
fig2.add_trace(go.Scatter(x=list(range(len(history.history['val_accuracy']))), 
                          y=history.history['val_accuracy'], 
                          mode='lines', 
                          name='Validierungsgenauigkeit'))

fig2.update_layout(title='Genauigkeit über Epochen',
                   xaxis_title='Epoche',
                   yaxis_title='Genauigkeit')

# Zeige die Plots
fig.show()
fig2.show()
