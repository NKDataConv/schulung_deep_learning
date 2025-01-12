"""
Übung: Lokales Deployment eines Modells in einem Docker-Container

Ziel:
In dieser Übung werden Sie ein einfaches Machine-Learning-Modell mithilfe von Scikit-Learn trainieren
und dieses Modell in einem Docker-Container bereitstellen. Der Container wird eine Flask-Webanwendung 
hosten, die Vorhersagen basierend auf Input-Daten trifft.

Schritte:
1. Trainieren und speichern Sie ein einfaches Machine-Learning-Modell mit Scikit-Learn.
2. Erstellen Sie eine Flask-Webanwendung, die folgendes bietet:
   - Empfängt Daten über eine POST-Anfrage.
   - Lädt das ML-Modell.
   - Gibt Vorhersagen auf Basis der empfangenen Daten zurück.
3. Erstellen Sie eine Docker-Datei, um die Flask-Anwendung in einem Container zu verpacken.
4. Bauen Sie das Docker-Image und führen Sie den Container lokal aus.
5. Testen Sie die bereitgestellte Anwendung mit Beispieldaten.

Das Python-Skript unten hilft Ihnen, den ersten Schritt zu erledigen: Erstellen und Speichern eines ML-Modells.
"""

# Importieren der benötigten Bibliotheken
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Laden des Iris-Datensatzes als Beispiel
iris = load_iris()
X, y = iris.data, iris.target

# Aufteilen des Datensatzes in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisieren des Modells
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Trainieren des Modells
model.fit(X_train, y_train)

# Evaluieren des Modells (optional, nur um das Training zu überprüfen)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Speichern des trainierten Modells in einer Datei
with open('iris_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

"""
Nächste Schritte:
- Verwenden Sie dieses Modell, um eine Flask-Anwendung zu erstellen, die es lädt und Webanfragen verarbeitet.
- Erstellen Sie eine 'Dockerfile', um die Anwendungsumgebung zu definieren:
  FROM python:3.8
  WORKDIR /app
  COPY requirements.txt requirements.txt
  RUN pip install -r requirements.txt
  COPY . .
  CMD ["python", "app.py"]
- Schreiben Sie die 'app.py'-Datei, um die Webanwendung zu implementieren.
- Stellen Sie sicher, dass 'requirements.txt' alle benötigten Abhängigkeiten wie Flask, scikit-learn usw. enthält.
- Bauen Sie das Docker-Image mit: docker build -t iris-flask-app .
- Führen Sie den Container aus mit: docker run -p 5000:5000 iris-flask-app
- Testen Sie die Anwendung mit HTTP-Anfragen.
"""
