### Erklärung zu den Code-Snippets:
#
# 1. **Softmax-Funktion**:
#    - Die Softmax-Funktion transformiert den Input-Vektor in Wahrscheinlichkeiten. Dabei wird die exponentielle Funktion verwendet, um sicherzustellen, dass die Werte nicht negativ sind und die Summe 1 ergibt, was als notwendige Bedingung für Wahrscheinlichkeiten gilt.
#
# 2. **Skalierte Dot-Produkt-Attention**:
#    - Dieser Teil ist das Herzstück des Attention-Mechanismus. Zunächst werden die Dot-Produkte zwischen den Abfragen(Q) und Schlüssel(K) berechnet.
#    - Danach wird das Ergebnis skaliert, um den Einfluss der Dimensionalität zu berücksichtigen und eine Überflutung der Werte zu vermeiden.
#    - Anschließend wird die Softmax-Funktion angewendet, um die Attention-Gewichtungen zu erhalten.
#    - Schließlich werden die Gewichtungen auf die Werte(V) angewandt, um das endgültige Ergebnis zu berechnen.

import numpy as np

def softmax(x):
    """
    Berechnet die Softmax-Funktion für einen Vektor.
    
    Args:
        x (numpy.ndarray): Input-Vektor.

    Returns:
        numpy.ndarray: Softmax-normalisierter Vektor.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Berechnet die Self-Attention mit skaliertem Dot-Produkt.
    
    Args:
        Q (numpy.ndarray): Abfrage-Matrix.
        K (numpy.ndarray): Schlüssel-Matrix.
        V (numpy.ndarray): Wert-Matrix.
        mask (numpy.ndarray, optional): Maske für die Attention.

    Returns:
        numpy.ndarray: Ergebnis der Attention-Gewichtung.
        numpy.ndarray: Attention-Gewichtungen.
    """
    # Berechnung der Dot-Produkte zwischen Abfragen (Q) und Schlüsseln (K)
    matmul_QK = np.dot(Q, K.T)
    
    # Berechnung der Dimension für die Skalierung
    dk = Q.shape[-1]  # Dimensionalität der Schlüssel
    scaled_attention_logits = matmul_QK / np.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # Setze maskierte Positionen auf einen sehr kleinen Wert
    
    # Berechnung der Softmax-Aktivierung
    attention_weights = softmax(scaled_attention_logits)
    
    # Berechnung der gewichteten Summe der Werte (V)
    output = np.dot(attention_weights, V)
    
    return output, attention_weights


# Beispiel-Input
# Hier nehmen wir an, dass wir 3 Sequenzen mit jeweils 4 Dimensionen haben
Q = np.array([[1, 0, 1, 0],
              [0, 1, 1, 0],
              [1, 1, 0, 0]])

K = np.array([[1, 0, 0, 1],
              [0, 1, 0, 1],
              [1, 1, 1, 0]])

V = np.array([[1, 1, 1, 0],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])

# Erstelle eine Maske, die die oberen Dreieckselemente ausschließt
mask = np.triu(np.ones((Q.shape[0], K.shape[0])), k=1)

# Berechnung der Self-Attention mit Maskierung
output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

# Ausgabe der Ergebnisse
print("Attention Weights:")
print(attention_weights)
print("\nOutput:")
print(output)
