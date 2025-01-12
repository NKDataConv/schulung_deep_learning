# TensorBoard zur Trainingsvisualisierung - Code Demo

# Zuerst müssen wir die benötigten Bibliotheken importieren
import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime

# Setzen von Seed für Reproduzierbarkeit
np.random.seed(42)
tf.random.set_seed(42)

# Generierung von Dummy-Daten für das Training
# Wir erstellen ein paar zufällige Daten für unsere Klassifikation
num_samples = 1000
num_features = 10
num_classes = 2

# Zufällige Merkmale
X_train = np.random.rand(num_samples, num_features).astype(np.float32)
# Zufällige Klassen (0 oder 1)
y_train = np.random.randint(num_classes, size=num_samples)

# Erstellen des Modells mit Keras
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(num_features,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

# Kompilieren des Modells mit Recall-Metrik
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 
                      tf.keras.metrics.Recall(class_id=1)])

# Erstellen eines Log-Verzeichnisses für TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Trainieren des Modells und gleichzeitig die TensorBoard-Callbacks nutzen
# Hier verwenden wir die Dummy-Daten, um das Demo-Skript einfach zu halten
model.fit(X_train, y_train, epochs=10, callbacks=[tensorboard_callback], validation_split=0.2)

# Nach dem Training kann TensorBoard gestartet werden, um die Ergebnisse zu visualisieren
# In der Kommandozeile folgendes ausführen, nachdem das Skript ausgeführt wurde:
# tensorboard --logdir=logs/fit

# Hinweis:
# 1. Öffne einen neuen Terminal-Tab oder -Fenster
# 2. Navigiere zum Verzeichnis, in dem dieses Skript gespeichert ist
# 3. Starte TensorBoard und benutze das angegebene Log-Verzeichnis
# 4. Gehe zu http://localhost:6006 für die Visualisierung

# TensorBoard bietet viele Möglichkeiten zur Visualisierung:
# - Verlustkurven
# - Genauigkeitskurven
# - Gewichts- und Aktivierungsverteilungen
# - Histograms von Layer-Gewichten und -Aktivierungen

# Wir können auch Modelldiagnosen und Hyperparameter-Optimierungen durchführen,
# indem wir weitere Metriken aufzeichnen und darstellen.
# TensorBoard ist ein wichtiges Werkzeug im MLOps-Toolkit, um das Training von Modellen zu überwachen,
# Probleme frühzeitig zu identifizieren und das Training zu optimieren.
