import numpy as np

# Ein Perceptron ist das einfachste Modell eines künstlichen Neurons.
# Es verarbeitet Eingabewerte, gewichtet sie und gibt das Ergebnis
# durch eine Aktivierungsfunktion aus.
# In dieser Demo implementieren wir ein einfaches Perceptron aus NumPy.

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        """
        Initialisiert das Perceptron mit zufälligen Gewichten und einer Lernrate.
        
        :param input_size: Anzahl der Eingabewerte
        :param learning_rate: Lernrate für das Training
        """
        self.weights = np.random.rand(input_size)  # Zufällige Initialisierung der Gewichte
        self.bias = np.random.rand(1)  # Bias zufällig initialisieren
        self.learning_rate = learning_rate  # Setzen der Lernrate

    def activation_function(self, x):
        """
        Aktivierungsfunktion, die das Ergebnis in 0 oder 1 umwandelt (Schwellwert).
        
        :param x: Eingabewert
        :return: 0 oder 1
        """
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        """
        Berechnet die Vorhersage für gegebene Eingaben.
        
        :param inputs: Eingabewerte
        :return: Vorhersage (0 oder 1)
        """
        weighted_sum = np.dot(inputs, self.weights) + self.bias  # Weighted Sum Calculation
        return self.activation_function(weighted_sum)  # Anwendung der Aktivierungsfunktion

    def train(self, training_inputs, labels, epochs):
        """
        Trainiert das Perceptron mit den gegebenen Eingaben und Labels.
        
        :param training_inputs: Liste der Eingabewerte für das Training
        :param labels: Gültige Ausgaben für die Eingabewerte
        :param epochs: Anzahl der Trainingsdurchläufe
        """
        for epoch in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)  # Vorhersage des Perceptrons
                # Fehlerberechnung
                error = label - prediction  
                # Aktualisierung der Gewichte und des Bias
                self.weights += self.learning_rate * error * inputs  
                self.bias += self.learning_rate * error  

# Anwendung der Code-Demo
if __name__ == "__main__":
    # Definieren von Eingabewerten und den zugehörigen Labels (XOR-Problem)
    # XOR ist ein klassisches Beispiel, das mit einem einfachen Perceptron nicht gelöst werden kann
    training_inputs = np.array([[0, 0],
                                 [0, 1],
                                 [1, 0],
                                 [1, 1]])
    labels = np.array([0, 1, 1, 0])  # XOR Labels

    # Initialisieren des Perceptrons mit 2 Eingängen (XOR hat 2 Eingabewerte)
    perceptron = Perceptron(input_size=2)

    # Trainieren des Perceptrons über 10.000 Epochen
    perceptron.train(training_inputs, labels, epochs=10000)

    # Testen des trainierten Perceptrons
    print("Testergebnisse:")
    for inputs in training_inputs:
        print(f"Eingaben: {inputs}, Vorhersage: {perceptron.predict(inputs)}")
