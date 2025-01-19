# Importiere notwendige Bibliotheken
import optuna
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tf_keras.models import Sequential
from tf_keras.layers import Dense
from tf_keras.optimizers import Adam

# Lade das Boston Housing Dataset als Beispiel
# Der Datensatz umfasst verschiedene Merkmale von Häusern in Boston, und wir wollen den Preis vorhersagen.
data = fetch_california_housing()
X = data.data
y = data.target

# Teilen der Daten in Trainings- und Testset
# Wir verwenden 50% der Daten für das Training und 50% für die Validierung, um das Training zu beschleunigen
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=42)

# Definiere die Bewertungsfunktion, die vom Optuna-Optimierer verwendet wird.
def objective(trial):
    # Optuna wählt die Hyperparameter basierend auf den definierten Bereichen
    n_layers = trial.suggest_int('n_layers', 1, 3)
    n_units = trial.suggest_int('n_units', 4, 128)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

    # Initialisiere das neuronale Netz
    model = Sequential()
    model.add(Dense(n_units, activation='relu', input_shape=(X_train.shape[1],)))
    for _ in range(n_layers - 1):
        model.add(Dense(n_units, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Kompiliere das Modell
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    # Trainiere das Modell
    print("Training model with n_layers={}, n_units={}, learning_rate={}".format(n_layers, n_units, learning_rate))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_data=(X_valid, y_valid))

    # Vorhersage auf dem Validierungs-Set
    preds = model.predict(X_valid).flatten()

    # Berechne die mittlere quadratische Abweichung (MSE) als Bewertungsmetrik
    mse = mean_squared_error(y_valid, preds)

    # Optuna minimiert die Funktion, daher geben wir die Metrik retour, die wir minimieren möchten
    return mse

# Erstelle ein Studienobjekt von Optuna, das die Optimierung durchführt
study = optuna.create_study(direction='minimize')

# Starte die Optimierung mit einer bestimmten Anzahl von Versuchen
study.optimize(objective, n_trials=10)

# Ausgabe der besten Hyperparameter und des besten Wertes aus der Optimierung
print("Beste Hyperparameter:")
print(study.best_params)
print("Beste MSE:", study.best_value)
