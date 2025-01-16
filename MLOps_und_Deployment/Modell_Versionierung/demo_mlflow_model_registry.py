# Vor dem Start muss die mlflow ui gestartet werden:
# im Terminal: mlflow ui

import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Constants
BATCH_SIZE = 32
EPOCHS = 5
MODEL_NAME = "MNISTModel"
ARTIFACT_LOCATION = "models"
EXPERIMENT_NAME = "MNISTModelExperiment"
TRACKING_URI = "http://127.0.0.1:5000"

# Set the MLflow Tracking Server
mlflow.set_tracking_uri(TRACKING_URI)

# Create an experiment in MLflow
mlflow.set_experiment(EXPERIMENT_NAME)

# Load and preprocess MNIST dataset
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape the data
X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# MLflow Tracking Block
with mlflow.start_run() as run:
    # Create a simple feedforward neural network
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    # Create progress bar for training
    print("\nTraining model...")

    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nModel Accuracy: {test_accuracy:.4f}")

    # Log parameters, metrics, and the trained model
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_metric("accuracy", test_accuracy)
    mlflow.log_metric("loss", test_loss)

    # Log the model in MLflow Model Registry
    mlflow.tensorflow.log_model(model, MODEL_NAME)

    # Show the Run ID
    print(f"Run ID: {run.info.run_id}")


# Register the model in MLflow Model Registry

from mlflow.tracking import MlflowClient

client = MlflowClient(TRACKING_URI)

# Add tags to the model
tags = {
    "model_type": "feedforward_nn",
    "dataset": "MNIST",
    "framework": "TensorFlow",
    "experiment_name": EXPERIMENT_NAME
}

model_version = client.create_registered_model(MODEL_NAME, tags=tags)

# Model Registration Integration
model_uri = f"runs:/{run.info.run_id}/{MODEL_NAME}"

# model_version = client.create_model_version(MODEL_NAME, model_uri, tags=tags)
print(f"Model registered with name: {model_version}")

# Load the model from registry
# model_from_registry = mlflow.tensorflow.load_model(model_uri)
model_from_registry = mlflow.tensorflow.load_model("MNISTModelTest")

# Make predictions with the registered model
predictions_from_registry = model_from_registry.predict(X_test[:5])
print(f"\nSample predictions from registered model: \n{np.argmax(predictions_from_registry, axis=1)}")
