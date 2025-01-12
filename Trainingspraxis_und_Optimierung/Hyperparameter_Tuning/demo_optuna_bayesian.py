# Importiere notwendige Bibliotheken
import optuna
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Lade das Boston Housing Dataset als Beispiel
# Der Datensatz umfasst verschiedene Merkmale von Häusern in Boston, und wir wollen den Preis vorhersagen.
data = load_boston()
X = data.data
y = data.target

# Teilen der Daten in Trainings- und Testset
# Wir verwenden 80% der Daten für das Training und 20% für die Validierung
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Definiere die Bewertungsfunktion, die vom Optuna-Optimierer verwendet wird.
def objective(trial):
    # Optuna wählt die Hyperparameter basierend auf den definierten Bereichen
    params = {
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', -1, 50),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0)
    }

    # Initialisiere das LightGBM-Modell mit den ausgewählten Hyperparametern
    model = lgb.LGBMRegressor(**params)

    # Trainiere das Modell
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], 
              eval_metric='rmse', early_stopping_rounds=50, verbose=False)

    # Vorhersage auf dem Validierungs-Set
    preds = model.predict(X_valid)

    # Berechne die mittlere quadratische Abweichung (MSE) als Bewertungsmetrik
    mse = mean_squared_error(y_valid, preds)

    # Optuna minimiert die Funktion, daher geben wir die Metrik retour, die wir minimieren möchten
    return mse

# Erstelle ein Studienobjekt von Optuna, das die Optimierung durchführt
study = optuna.create_study(direction='minimize')

# Starte die Optimierung mit einer bestimmten Anzahl von Versuchen
study.optimize(objective, n_trials=100)

# Ausgabe der besten Hyperparameter und des besten Wertes aus der Optimierung
print("Beste Hyperparameter:")
print(study.best_params)
print("Beste MSE:", study.best_value)

# In diesem Skript haben wir eine vollständige Code-Demo für Hyperparameter-Tuning mit Optuna erstellt, insbesondere für ein LightGBM-Modell, das auf dem Boston Housing Dataset basiert.

### Wichtige Punkte aus dem Kapitel 'Hyperparameter Tuning'

# 1. **Hyperparameter**: Dies sind Parameter des Modells, die nicht aus den Trainingsdaten gelernt werden, sondern vor dem Training definiert werden müssen (z.B. Lernrate, Anzahl der Blätter usw.).
#
# 2. **Optimierung**: In diesem Skript verwenden wir die Bayesian Optimization, welche effizientere Hyperparameter-Kombinationen liefert, indem sie den Suchraum intelligent erkundet.
#
# 3. **Bewertungsfunktion**: Hier definieren wir die `objective()`-Funktion, die Optuna verwendet, um den besten Satz an Hyperparametern zu finden, basierend auf der Metrik `mean_squared_error` (MSE).
#
# 4. **Studie durchführen**: Über `study.optimize()` führen wir die Optimierung für eine definierte Anzahl von Trials durch (hier 100).
#
# Die Demo zeigt, wie man Hyperparameter in einem Machine Learning-Modell optimieren kann – ein entscheidender Schritt in der Praxis der Modellentwicklung.