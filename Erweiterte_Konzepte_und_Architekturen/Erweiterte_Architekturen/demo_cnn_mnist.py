# Importieren der notwendigen Bibliotheken
import plotly.express as px
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Laden der MNIST-Daten (Handschriftliche Ziffern)
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Die MNIST-Daten bestehen aus 28x28 Graustufenbildern
# Visualisierung der ersten 9 Bilder im Datensatz

# fig = px.imshow(train_images[:9], facet_col=0, facet_col_wrap=3, labels={'x': 'Ziffer', 'y': 'Pixel'}, title='Erste 9 Bilder im MNIST-Datensatz')
# fig.update_traces(text=train_labels[:9])
# fig.show()

# Vorverarbeitung der Daten
# Die Bilder werden normalisiert (Werte zwischen 0 und 1), um die Trainingsstabilität zu erhöhen
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# One-Hot-Encoding der Labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Aufbau des Convolutional Neural Networks (CNN)
model = models.Sequential()

# Erste Convolutional-Schicht mit 32 Filtern und 3x3 Kernel
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# Max-Pooling-Schicht zur Reduzierung der Dimensionsanzahl
model.add(layers.MaxPooling2D((2, 2)))

# Zweite Convolutional-Schicht mit 64 Filtern
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Zweite Max-Pooling-Schicht
model.add(layers.MaxPooling2D((2, 2)))

# Dritte Convolutional-Schicht mit 64 Filtern
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten-layer konvertiert die 3D-Ausgabe in eine 1D-Ausgabe
model.add(layers.Flatten())
# Vollständig verbundene Schicht
model.add(layers.Dense(64, activation='relu'))
# Ausgabeschicht mit 10 Neuronen (für 10 Klassen)
model.add(layers.Dense(10, activation='softmax'))

# Kompilierung des Modells
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Erstellen des TensorBoard-Log-Verzeichnisses
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# TensorBoard-Callback hinzufügen
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training des Modells
# Weitere Parameter:
# epochs: Anzahl der Durchläufe durch den gesamten Datensatz
# batch_size: Anzahl der Bilder, die in einer Iteration verarbeitet werden
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])

# Auswertung des Modells
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc:.4f}')

# Visualisierung der Trainingshistorie
fig_accuracy = px.line(x=list(range(1, 6)), y=history.history['accuracy'], labels={'x': 'Epochs', 'y': 'Accuracy'}, title='Accuracy over epochs')
fig_accuracy.add_scatter(x=list(range(1, 6)), y=history.history['val_accuracy'], mode='lines', name='Valid accuracy')
fig_accuracy.show()

fig_loss = px.line(x=list(range(1, 6)), y=history.history['loss'], labels={'x': 'Epochs', 'y': 'Loss'}, title='Loss over epochs')
fig_loss.add_scatter(x=list(range(1, 6)), y=history.history['val_loss'], mode='lines', name='Valid loss')
fig_loss.show()
