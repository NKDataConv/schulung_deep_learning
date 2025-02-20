# Anmerkungen zur GPU-Nutzung und Mixed Precision
'''
Mixed Precision Training nutzt weniger Speicherplatz und kann das Training auf modernen GPUs beschleunigen.
Es kombiniert float16 und float32, um Rechenleistung zu optimieren.
- dtype: 'float16' für die Layer und Variablen, wo es sinnvoll ist
- Der Optimizer und der Verlust beinhalten weiterhin float32 zur Vermeidung von Unterlauf/Überlauf-Problemen.

Vor der Nutzung von Mixed Precision Training, sollte sichergestellt werden, dass die Hardware (z.B. NVIDIA GPUs) dies unterstützt und die TensorFlow-Version aktuell ist.
'''

import os

# Deaktivieren der GPU-Nutzung
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Importieren der notwendigen Bibliotheken
import tensorflow as tf
from tf_keras import layers, models
from tf_keras import mixed_precision
import numpy as np

# Sicherstellen, dass nur die CPU verwendet wird
tf.config.set_visible_devices([], 'GPU')

# Überprüfung, ob eine GPU vorhanden ist und ob Mixed Precision unterstützt wird
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"Es wurde eine GPU gefunden: {gpus[0]}")
else:
    print("Keine GPU gefunden, das Training wird auf der CPU durchgeführt.")

# Aktivieren von Mixed Precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed Precision Policy gesetzt auf: {policy.name}")

# Erstellen von Beispiel-Daten
# Simulieren von 1000 Sätzen, jeder mit 50 Wörtern (dies ist ein Beispiel, um das Skript zu veranschaulichen)
num_samples = 1000
max_length = 50
vocab_size = 2000

# Zufällige Eingabedaten
X_train = np.random.randint(1, vocab_size, size=(num_samples, max_length))
# Zielwerte setzen (0 oder 1 für binäre Klassifizierung)
y_train = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float16)

def create_model():
    """
    Creates and returns a RNN model with mixed precision
    """
    model = models.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length, dtype='float16'),
        layers.SimpleRNN(64, return_sequences=True, dtype='float16'),
        layers.SimpleRNN(32, dtype='float16'),
        layers.Dense(1, activation='sigmoid', dtype='float16')
    ])
    
    model.compile(optimizer='adam', 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    
    return model

# Create and use the model
model = create_model()
model.summary()

# Trainieren des Modells
history = model.fit(X_train, y_train, epochs=10, batch_size=32)

# Bewertung des Modells
loss, accuracy = model.evaluate(X_train, y_train)
print(f"Modell Verlust: {loss:.4f}, Genauigkeit: {accuracy:.4f}")
