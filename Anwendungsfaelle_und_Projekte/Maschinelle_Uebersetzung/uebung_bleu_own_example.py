# Aufgabenstellung für die Schulungsteilnehmer:
# Übung: BLEU-Werte für ein eigenes Übersetzungsbeispiel berechnen

# Ziel: Es soll ein Python Skript erstellt werden, das den BLEU-Wert für einen maschinell übersetzten Text
#       berechnet. Hierbei wird die Referenzübersetzung manuell bereitgestellt.
# Schritte:
# 1. Wählen Sie eine einfache englische Textpassage bestehend aus einem Satz.
# 2. Schreiben Sie eine eigene deutsche Übersetzung (Referenzübersetzung) dieses Satzes.
# 3. Erstellen Sie eine maschinell generierte Übersetzung desselben Satzes.
# 4. Bewertern Sie die Qualität der Übersetzung mit dem BLEU-Score, indem Sie die Referenzübersetzung
#    mit der maschinell generierten Übersetzung vergleichen.
#
# Für die Implementierung sollen Sie:
# - Die Bibliothek nltk nutzen, um den BLEU-Score zu berechnen
# - Mindestens einen BLEU-Score (1-gram) berechnen
# - Ihre Ergebnisse ausdrucken

# Importieren der notwendigen Bibliotheken
from nltk.translate.bleu_score import sentence_bleu

# Schritt 1: Englischen Ausgangstext definieren (zu analysierender Satz)
original_text = "The quick brown fox jumps over the lazy dog."

# Schritt 2: Manuelle Referenzübersetzung (deutsche) definieren
reference_translation = "Der schnelle braune Fuchs springt über den faulen Hund."

# Schritt 3: Maschinell generierte Übersetzung definieren
# Zum Beispiel könnte eine Übersetzungsmaschine folgenden Satz generieren:
machine_translation = "Der schnelle braune Fuchs springt über den faulen Hund."

# Schritt 4: Berechnen Sie den BLEU-Wert

# NLTK erwartet eine Liste von Referenzübersetzungen, da es in der Praxis mehrere geben kann.
# Hier haben wir nur eine Referenzübersetzung, deshalb legen wir sie in einer Liste der Listen an.
references = [reference_translation.split()]

# Die maschinell generierte Übersetzung muss ebenfalls in eine Liste von Wörtern aufgeteilt werden
candidate = machine_translation.split()

# Berechnen des BLEU-Scores, hier verwenden wir 1-gram BLEU score
# Der BLEU-Score reicht von 0 bis 1, wobei 1 eine perfekte Übereinstimmung bedeutet.
bleu_score = sentence_bleu(references, candidate, weights=(1, 0, 0, 0))

# Ausgeben des BLEU-Werts
print(f"Der 1-gram BLEU-Score für die Übersetzung ist: {bleu_score:.2f}")

# Hinweis: In diesem Beispiel ist der BLEU-Score 1, da die beiden Übersetzungen identisch sind.
