# Aufgabe: Canary Deployment für ein aktualisiertes Modell durchführen
#
# Beschreibung:
# In dieser Übung werden Sie ein einfaches Canary Deployment für ein aktualisiertes Maschinelles Lernmodell durchführen. 
# Ein Canary Deployment ist eine Technik, um ein neues Software-Update (in diesem Fall ein ML-Modell) schrittweise einzuführen. 
# Ziel ist es, die neue Version nur einem kleinen Teil der Nutzer zur Verfügung zu stellen und ihre Leistung zu überwachen, 
# bevor sie vollständig eingeführt wird.
#
# Voraussetzungen:
# - Ein bestehendes Modell (modell_1) in Produktion
# - Ein neues, aktualisiertes Modell (modell_2), das eingeführt werden soll
# - Eine Methode, um Anfragen zu den Modellen zu leiten
# - Monitoring der Leistungskennzahlen, um die beiden Modelle zu vergleichen
#
# Ziele:
# - Implementieren Sie einen einfachen Canary Deployment-Mechanismus mit Python.
# - Führen Sie die Modelle in einer simulierten Umgebung aus.
# - Berechnen Sie die Leistungskennzahlen beider Modelle.

# Schritte:
# 1. Erstellen Sie zwei Beispielmodelle.
# 2. Implementieren Sie eine Klasse für das Canary Deployment.
# 3. Verwalten Sie die Anfragen so, dass ein Teil zu modell_1 und ein Teil zu modell_2 geht.
# 4. Vergleichen Sie die Leistung der beiden Modelle anhand von Testdaten.

import random
from sklearn.metrics import accuracy_score

# Schritt 1: Erstellen von Beispieldaten und Modellen
# Annahme: Wir haben zwei sehr einfache Modelle, die zufällige Vorhersagen treffen.

def create_model_1():
    """Stellt das bestehende Modell dar. Macht zufällige Vorhersagen."""
    return lambda x: random.choice([0, 1])

def create_model_2():
    """Stellt das neue, aktualisierte Modell dar. Macht zufällige Vorhersagen."""
    return lambda x: random.choice([0, 1])

# Schritt 2: Implementieren einer Klasse für das Canary Deployment
class CanaryDeployment:
    def __init__(self, model_1, model_2, canary_percentage):
        """
        Initialisiert die Canary Deployment Klasse.
        
        :param model_1: Funktion des ersten Modells (bestehendes Modell).
        :param model_2: Funktion des zweiten Modells (aktualisiertes Modell).
        :param canary_percentage: Prozentsatz der Anfragen, die an das neue Modell geleitet werden.
        """
        self.model_1 = model_1
        self.model_2 = model_2
        self.canary_percentage = canary_percentage

    def make_prediction(self, data):
        """
        Entscheidet basierend auf `canary_percentage`, welches Modell verwendet wird 
        und gibt eine Vorhersage aus.
        
        :param data: Die Eingabedaten für die Vorhersage.
        :return: Vorhersageergebnis (0 oder 1).
        """
        if random.random() < self.canary_percentage:
            return self.model_2(data)
        else:
            return self.model_1(data)

# Schritt 3: Simulieren von Anfragen und Modellwahl
# Testen Sie das Canary Deployment in einer simulierten Umgebung.

# Erstellen der Modelle
model_1 = create_model_1()
model_2 = create_model_2()

# Erstellen des Canary Deployment Mechanismus
canary_deployment = CanaryDeployment(model_1, model_2, canary_percentage=0.1)

# Schritt 4: Vergleich der Leistung
# Generieren von Testdaten
test_data = [random.randint(0, 100) for _ in range(1000)]
actual_labels = [random.choice([0, 1]) for _ in range(1000)]

# Sammeln von Vorhersagen und Evaluierung der Modelle
predictions_1 = []
predictions_2 = []

for data in test_data:
    pred = canary_deployment.make_prediction(data)
    if pred == model_1(data):
        predictions_1.append(pred)
    else:
        predictions_2.append(pred)

# Berechnung der Genauigkeit der Vorhersagen für jede Modellgruppe
accuracy_model_1 = accuracy_score(actual_labels, [model_1(data) for data in test_data])
accuracy_model_2 = accuracy_score(actual_labels, [model_2(data) for data in test_data])

print(f"Genauigkeit des bestehenden Modells 1: {accuracy_model_1:.2f}")
print(f"Genauigkeit des aktualisierten Modells 2: {accuracy_model_2:.2f}")

# Hinweis: In einem echten Szenario sollten auch andere Metriken betrachtet werden, 
# und Daten sollten gleichmäßig auf beide Modelle verteilt werden, um faire Vergleichsgrundlagen zu schaffen.
