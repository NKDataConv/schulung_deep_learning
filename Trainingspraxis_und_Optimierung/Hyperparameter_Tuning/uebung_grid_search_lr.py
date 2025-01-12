# Aufgabe: Übung - Grid Search auf die Learning Rate anwenden
# In dieser Übung werden Sie die Technik des Hyperparameter-Tunings verwenden, um die optimale 
# Lernrate (Learning Rate) für ein neuronales Netzwerk in einem Textklassifizierungsproblem zu finden.
# Sie werden Grid Search verwenden, um verschiedene Werte der Lernrate zu testen und die beste Konfiguration 
# mit Hilfe eines Validierungsdatensatzes zu bestimmen.

# 1. Laden Sie einen Textdatensatz, zum Beispiel den '20 Newsgroups' Datensatz.
# 2. Bereiten Sie die Texte für die Verarbeitung vor, indem Sie sie in numerische Form umwandeln (z.B. mit 
# CountVectorizer oder TfidfVectorizer).
# 3. Implementieren Sie ein einfaches neuronales Netzwerk mit Keras/TensorFlow, das für die Klassifizierung 
# der Daten verwendet wird.
# 4. Führen Sie eine Grid Search durch, um die beste Lernrate aus einer vorgegebenen Liste von Lernraten zu 
# finden. Nutzen Sie dabei KFold-Cross-Validation.
# 5. Geben Sie die beste Lernrate aus und das zugehörige Modell-Score auf dem Validierungsdatensatz.
# 6. Kommentieren Sie Ihren Code ausführlich, um jeden Schritt der Übung zu erklären.

# Beginnen wir mit der Implementierung:

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import make_pipeline
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
import numpy as np

# Funktion, die unser Modell erstellt
def create_model(learning_rate=0.01):
    # Initialisieren des Keras Sequential Modells
    model = Sequential()
    # Hinzufügen der Eingangsschicht mit 512 Neuronen und ReLU Aktivierungsfunktion
    model.add(Dense(units=512, activation='relu', input_shape=(num_features,)))
    # Hinzufügen einer Dropout-Schicht zur Vermeidung von Überanpassung
    model.add(Dropout(0.5))
    # Hinzufügen der Ausgangsschicht, die so viele Outputs wie Klassen hat, und softmax Aktivierungsfunktion
    model.add(Dense(units=num_classes, activation='softmax'))
    # Konfigurieren des Optimizers mit der aktuellen Lernrate
    optimizer = Adam(learning_rate=learning_rate)
    # Kompilieren des Modells mit der Verlustfunktion 'categorical_crossentropy' und dem gewählten Optimizer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Laden des 20 Newsgroups Datensatzes
newsgroups_train = fetch_20newsgroups(subset='train', categories=None, shuffle=True, random_state=42)

# Konvertieren von Textdaten in TF-IDF Merkmalsdarstellung
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(newsgroups_train.data)

# Binarisieren der Labels für unsere Klassifizierung
lb = LabelBinarizer()
y = lb.fit_transform(newsgroups_train.target)

# Festlegen der Anzahl der Merkmale und Klassen für das Modell
num_features = X.shape[1]  # Anzahl der Merkmale
num_classes = len(lb.classes_)  # Anzahl der zu klassifizierenden Klassen

# Mögliche Werte für die Lernrate bei der Grid Search
learning_rates = [0.001, 0.01, 0.1, 0.2]

# KFold-Cross-Validation einstellen
cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Beste Modellbewertung initialisieren
best_score = 0
best_lr = None

# Durchlaufen aller Lernraten und Durchführung der Grid Search
for lr in learning_rates:
    print(f"Training mit Learning Rate: {lr}")
    # Keras Modell in einen scikit-learn kompatiblen KerasClassifier umwandeln
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=128, learning_rate=lr, verbose=0)
    # Pipeline erstellen, um die Datenverarbeitungsschritte zusammen mit dem Modell zu handhaben
    pipeline = make_pipeline(vectorizer, model)
    # Kreuzvalidierung verwenden, um die durchschnittliche Leistung der aktuellen Lernrate zu messen
    scores = cross_val_score(pipeline, newsgroups_train.data, y, cv=cv, scoring='accuracy')
    # Mittlerer Score berechnet
    mean_score = np.mean(scores)
    print(f"Mean Validation Accuracy for learning rate {lr}: {mean_score}")
    # Aktualisierung der besten gefundenen Lernrate und Bewertung
    if mean_score > best_score:
        best_score = mean_score
        best_lr = lr

# Ausgabe der besten Lernrate
print(f"Beste Learning Rate: {best_lr} mit einem Score von: {best_score}")
