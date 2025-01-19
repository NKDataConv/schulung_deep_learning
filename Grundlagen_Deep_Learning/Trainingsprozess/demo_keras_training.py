# Importieren der notwendigen Bibliotheken
import numpy as np
from tf_keras.models import Sequential
from tf_keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

# Beispieldaten generieren: Simulieren von Merkmalen und Zielvariablen
# Hier verwenden wir ein einfaches Dataset mit 1000 Proben.
np.random.seed(42)
X = np.random.rand(1000, 10)  # 10 Merkmale
y = (np.sum(X, axis=1) > 5).astype(int)  # Zielvariable: 1 wenn die Summe der Merkmale > 5, sonst 0

# Daten in Trainings- und Testdatensätze aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Daten normalisieren: Standardisierung der Merkmale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def create_model():
    """
    Creates and returns a simple neural network model
    """
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))  # Input layer + hidden layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    
    model.compile(loss='binary_crossentropy',  # Loss function for binary classification
                 optimizer='adam',             # Adam optimizer
                 metrics=['accuracy'])         # Metric for performance evaluation
    
    return model

# Create and train the model
model = create_model()
model.summary()

# Training des Modells
# Wir verwenden die fit-Methode, um das Modell mit Trainingsdaten zu trainieren
history = model.fit(X_train, y_train, 
                    epochs=20,     # Anzahl der Epochen (Durchläufe über das gesamte Dataset)
                    batch_size=16, # Anzahl der Samples pro Gradientenaktualisierung
                    validation_data=(X_test, y_test),  # Validierungsdaten
                    verbose=1)     # Fortschrittsausgabe während des Trainings anzeigen

# Modellbewertung
# Vorhersagen auf den Testdaten
y_pred = (model.predict(X_test) > 0.5).astype(int) # Vorhersage mit Schwellwert von 0.5
# Berechnung der Genauigkeit
accuracy = accuracy_score(y_test, y_pred)
print(f"Testgenauigkeit: {accuracy:.2f}")

# Visualisierung der Trainingsverluste und -genauigkeit

# Verlust und Genauigkeit während des Trainings plotten
fig = go.Figure()

# Verlust plotten
fig.add_trace(go.Scatter(x=list(range(1, 51)), y=history.history['loss'], mode='lines', name='Trainingsverlust'))
fig.add_trace(go.Scatter(x=list(range(1, 51)), y=history.history['val_loss'], mode='lines', name='Validierungsverlust'))

# Genauigkeit plotten
fig.add_trace(go.Scatter(x=list(range(1, 51)), y=history.history['accuracy'], mode='lines', name='Trainingsgenauigkeit'))
fig.add_trace(go.Scatter(x=list(range(1, 51)), y=history.history['val_accuracy'], mode='lines', name='Validierungsgenauigkeit'))

# Layout anpassen
fig.update_layout(title='Verlust und Genauigkeit während des Trainings',
                  xaxis_title='Epochen',
                  yaxis_title='Wert',
                  legend_title='Legende')

# Plot anzeigen
fig.show()
