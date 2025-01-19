# **Erklärung des Skripts:**
# 1. **Imports:** TensorFlow und NumPy werden importiert, um das Modell zu erstellen und mit Daten zu arbeiten.
# 2. **Daten:** Beispieltexte und ihre zugehörigen Labels werden definiert.
# 3. **Hyperparameter:** Batch-Größe und AUTOTUNE für eine optimierte Datensatzbearbeitung sind festgelegt.
# 4. **Textvorverarbeitung:** Eine Funktion `preprocess_text` fördert die Standardisierung von Texten.
# 5. **Datenpipeline:** `create_dataset` erstellt eine Datensatz-Pipeline unter Verwendung von `tf.data`, die Vorverarbeitung, Mischen und Batching enthält.
# 6. **Modellerstellung:** Ein einfaches neuronales Netzwerk wird definiert, das Textdaten verarbeitet.
# 7. **Trainingsprozess:** Das Modell wird kompiliert und mit den Trainingsdaten trainiert.
# 8. **Vorhersagefunktion:** Eine Funktion ermöglicht die Vorhersage für neue Texte.
# 9. **Beispielvorhersage:** Eine Beispielvorhersage wird durchgeführt und das Ergebnis wird ausgegeben.


import tensorflow as tf
import numpy as np

# Hyperparameter
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 5

# Lade den MNIST-Datensatz
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalisiere die Bilder
train_images = train_images / 255.0
test_images = test_images / 255.0

# Erstelle TensorFlow Dataset aus den Bildern und Labels
def create_dataset(images, labels):
    # Erstelle ein tf.data Dataset
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    # Mische die Daten und teile sie in Batches auf
    data = data.shuffle(buffer_size=len(images)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return data

# Bereite die Trainings- und Testdaten vor
train_dataset = create_dataset(train_images, train_labels)
test_dataset = create_dataset(test_images, test_labels)

def create_model():
    """
    Creates and returns a simple neural network for MNIST classification
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

# Create and use the model
model = create_model()

# Trainiere das Modell
model.fit(train_dataset, epochs=EPOCHS)

# Evaluieren Sie das Modell
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")

# Demonstration der Vorhersage
def predict(image):
    # Erweitere die Dimensionen des Bildes für die Vorhersage
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return np.argmax(prediction, axis=1)

# Beispielvorhersage
input_image = test_images[0]
predicted_label = predict(input_image)

print(f"Vorhersage für das erste Testbild: {predicted_label[0]}")
