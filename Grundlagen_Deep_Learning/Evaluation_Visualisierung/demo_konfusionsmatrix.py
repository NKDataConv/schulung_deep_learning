import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout
from tf_keras.callbacks import TensorBoard
import plotly.figure_factory as ff

# Konstanten
RANDOM_SEED = 42
TEST_SIZE = 0.3
EPOCHS = 50
BATCH_SIZE = 32
LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Laden des Iris-Datensatzes
iris = load_iris()
X = iris.data
y = iris.target

# Binäre Klassifikation: Setosa vs. Rest
y_binary = (y == 0).astype(int)

# Daten normalisieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED
)

def create_model():
    """
    Creates and returns a simple neural network for binary classification
    """
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(4,)))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

print("Building neural network...")
model = create_model()

# TensorBoard Callback mit Confusion Matrix
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Alle 5 Epochen
            predictions = self.model.predict(X_test)
            predictions = (predictions > 0.5).astype(int)
            cm = tf.math.confusion_matrix(y_test, predictions).numpy()

            # Erstellen der Konfusionsmatrix mit Plotly
            labels = ['Not Setosa', 'Setosa']
            figure = ff.create_annotated_heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale='Blues',
                showscale=True
            )

            figure.update_layout(
                title=f'Confusion Matrix - Epoch {epoch}',
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                width=800,
                height=800
            )

            # Convert Plotly figure to image
            img_bytes = figure.to_image(format="png")

            # Log to TensorBoard
            with tf.summary.create_file_writer(LOG_DIR).as_default():
                image = tf.image.decode_png(img_bytes, channels=4)
                image = tf.expand_dims(image, 0)
                tf.summary.image("Confusion Matrix", image, step=epoch)

print("Setting up TensorBoard...")
# TensorBoard Callbacks
tensorboard_callback = TensorBoard(
    log_dir=LOG_DIR,
    histogram_freq=1,
    write_graph=True
)

cm_callback = ConfusionMatrixCallback()

# Model Training
model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[
        tensorboard_callback,
        cm_callback
    ],
    verbose=0
)

print("\nTraining completed! To view results:")
print(f"Run: tensorboard --logdir {LOG_DIR}")
print("Then open http://localhost:6006 in your browser")
