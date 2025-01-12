# Import der benötigten Bibliotheken
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Setze den MLflow Tracking Server (falls lokal, kann auch mit einem Remote Server ersetzt werden)
mlflow.set_tracking_uri("http://localhost:5000")  # Beispiel URI für den Tracking Server

# Projektkonfiguration
model_name = "IrisRandomForestModel"
artifact_location = "models"  # Speicherort der Modelle
experiment_name = "IrisModelExperiment"  # Experimentname

# Erstelle ein Experiment in MLflow
mlflow.set_experiment(experiment_name)

# Lade den Iris-Datensatz
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Teile die Daten in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow Tracking Block
with mlflow.start_run() as run:
    # Trainiere ein Random Forest Modell
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Mache Vorhersagen auf dem Test-Set
    predictions = model.predict(X_test)

    # Berechne die Genauigkeit des Modells
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Logge Hyperparameter, Metriken und das trainierte Modell
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)

    # Logge das Modell in den MLflow Model Registry
    mlflow.sklearn.log_model(model, model_name)

    # Zeige die Run-ID für den weiteren Einsatz und Referenz
    print(f"Run ID: {run.info.run_id}")

# Model Registrierungsintegration
# Hier könnten wir das Modell registrieren, um es in einer CI/CD-Pipeline verfügbar zu machen
model_uri = f"runs:/{run.info.run_id}/{model_name}"

# Registriere das Modell in der MLflow Model Registry
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_version = client.create_registered_model(model_name)
print(f"Model registered with name: {model_version}")

# Prototyp (konzeptionell, nicht implementiert) für CI/CD, wo wir dieses Modell möglicherweise deployen möchten
# Schritt 1: CI/CD Tool konfigurieren (z.B. Jenkins, GitHub Actions)
# Schritt 2: Modell mit der MLflow API abrufen und in einer Produktionsumgebung bereitstellen

# Um das Modell aus der Registry abzurufen
registered_model_uri = f"models:/{model_name}/1"  # '1' ist eine Beispielversion
model_from_registry = mlflow.sklearn.load_model(registered_model_uri)

# Mache Vorhersagen mit dem registrierten Modell
predictions_from_registry = model_from_registry.predict(X_test)
print(f"Predictions from registered model: {predictions_from_registry}")

# Speicher das Modell unsere eigene .pkl Datei für die zukunft
joblib.dump(model, 'RandomForestModel.pkl')

# Hinweis: In einer echten CI/CD Pipeline können Schritte hinzugefügt werden,
# um automatisierte Tests, Modellüberwachung, etc. zu integrieren.
