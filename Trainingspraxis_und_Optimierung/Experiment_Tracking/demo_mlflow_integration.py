### Erläuterung der wichtigsten Elemente:

# 1. **MLflow-Session**: Die Skript-Session wird mit `mlflow.start_run()` gestartet und mit `mlflow.end_run()` beendet, um einen zusammenhängenden Tracking-Vorgang zu schaffen.
#
# 2. **Datensatz**: Der Iris-Datensatz wird als Beispiel verwendet, um grundlegende Klassifizierungsaufgaben zu demonstrieren.
#
# 3. **Modelltraining**: Es wird ein `RandomForestClassifier` trainiert, und die Vorhersagen sowie die Genauigkeit werden erhalten.
#
# 4. **Hyperparameter- und Metrik-Logging**: Wichtige Kennzahlen und Hyperparameter werden in MLflow geloggt, was eine effektive Nachverfolgbarkeit der Experimente ermöglicht.
#
# 5. **Modell-Speicherung**: Das trainierte Modell wird gespeichert, sodass es später verwendet oder in die Produktion integriert werden kann.
# 
# 6. Starte die UI mit `mlflow ui` (im Terminal)

# Importieren der notwendigen Bibliotheken
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Starten einer MLflow-Session
mlflow.start_run()

# Schritt 1: Laden der Daten
print("Laden der Iris-Daten...")
iris = load_iris()
X, y = iris.data, iris.target

# Schritt 2: Aufteilen der Daten in Trainings- und Testdaten
print("Aufteilen der Daten in Trainings- und Testdaten...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Schritt 3: Training des Modells
print("Training des Random Forest Classifiers...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Schritt 4: Vorhersagen treffen
print("Treffen der Vorhersagen auf den Testdaten...")
y_pred = model.predict(X_test)

# Schritt 5: Evaluierung des Modells
print("Evaluierung des Modells...")
accuracy = accuracy_score(y_test, y_pred)
print(f"Modellgenauigkeit: {accuracy:.2f}")
print("Klassifikationsbericht:")
print(classification_report(y_test, y_pred))

# Schritt 6: Logging der Hyperparameter und Metriken mit MLflow
print("Logging der Hyperparameter und Metriken mit MLflow...")
# Logging der Hyperparameter
mlflow.log_param("n_estimators", 100)
mlflow.log_param("random_state", 42)

# Logging der Metriken
mlflow.log_metric("accuracy", accuracy)

# Schritt 7: Speichern des Modells mit einem Input-Beispiel
print("Speichern des Modells mit MLflow...")
input_example = X_test[0:1]  # Verwenden Sie ein Beispiel aus den Testdaten
mlflow.sklearn.log_model(model, "random_forest_model", input_example=input_example)

# Beenden der MLflow-Session
mlflow.end_run()

print("Die MLflow-Integration in den Trainings-Workflow wurde erfolgreich durchgeführt.")
print("Sie können die Ergebnisse im MLflow UI überprüfen.")
