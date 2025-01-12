"""
Aufgabenstellung:
In dieser Übung werden Sie ein einfaches Machine Learning Modell erstellen, es versionieren und im MLflow Model Registry 
speichern. MLflow ist ein Open-Source Plattform zur Verwaltung des gesamten Maschinenlernlebenszyklus. 
Sie werden folgende Schritte durchführen:

1. Erstellen Sie ein einfaches lineares Regressionsmodell mithilfe von Scikit-Learn.
2. Trainieren Sie das Modell mit dem bekannten 'Boston Housing' Datensatz.
3. Richten Sie MLflow ein und verwenden Sie es, um das trainierte Modell mit einem eindeutigen Tag zu speichern.
4. Registrieren Sie das gespeicherte Modell in der MLflow Model Registry.
5. Überprüfen Sie die Modellversionen und heben Sie die Details über den registrierten Modellß

Stellen Sie sicher, dass Sie MLflow installiert haben (`pip install mlflow`).
"""

# Import necessary libraries
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Boston Housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate and print the Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error: ", mse)

# Start an MLflow run to log parameters and metrics
with mlflow.start_run():
    # Log the model
    mlflow.sklearn.log_model(model, "linear_regression_model")
    
    # Log the mean squared error as a metric
    mlflow.log_metric("mse", mse)
    
    # Create a tag for the model
    mlflow.set_tag("version", "1.0")
    mlflow.set_tag("stage", "Development")
    
    # Register the model in the MLflow Model Registry
    model_uri = "runs:/{}/linear_regression_model".format(mlflow.active_run().info.run_id)
    model_details = mlflow.register_model(model_uri, "BostonHousingModel")
    
    # Move the model to the "staging" stage
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage="Production",
        archive_existing_versions=False
    )

    # Print model details
    print("Model registered with name {} and version {}".format(model_details.name, model_details.version))
