# Code Demo: Berechnung von BLEU und ROUGE für ein Übersetzungsmodell

# Wir verwenden zur Berechnung der Metriken das NLTK-Paket für BLEU und das rouge-score-Paket für ROUGE.
# Achten Sie darauf, dass die notwendigen Pakete installiert sind:
# pip install nltk rouge-score

import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# NLTK für BLEU nutzen
nltk.download('punkt')

# Funktion zur Berechnung von BLEU
def calculate_bleu(reference, translation):
    """
    Berechnet den BLEU-Score für die gegebene Übersetzung.
    
    :param reference: Liste von Referenzsätzen (Liste von Listen)
    :param translation: Übersetzung als Liste von Tokens
    :return: BLEU-Score
    """
    # Berechne den BLEU-Score
    bleu_score = sentence_bleu(reference, translation)
    return bleu_score

# Funktion zur Berechnung von ROUGE
def calculate_rouge(reference, translation):
    """
    Berechnet den ROUGE-Score für die gegebene Übersetzung.
    
    :param reference: Referenzsatz
    :param translation: Übersetzung
    :return: ROUGE-Score
    """
    # Initialisieren des ROUGE Scorers
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, translation)
    return scores

# Hauptfunktion zur Durchführung der Metrikberechnung
def main():
    # Beispielreferenz und Übersetzung
    reference_sentences = [["this", "is", "a", "sample", "reference", "sentence"],
                           ["this", "is", "another", "example"]]
    translation_sentence = ["this", "is", "a", "sample", "translated", "sentence"]

    # Berechnung des BLEU-Scores
    bleu_score = calculate_bleu(reference_sentences, translation_sentence)
    print(f"BLEU-Score: {bleu_score:.4f}")

    # Für ROUGE benötigen wir einfache Sätze (Strings) anstatt Listen
    reference_sentence = "this is a sample reference sentence"
    translation_sentence_str = "this is a sample translated sentence"

    # Berechnung des ROUGE-Scores
    rouge_scores = calculate_rouge(reference_sentence, translation_sentence_str)
    print(f"ROUGE-1: {rouge_scores['rouge1']}")
    print(f"ROUGE-2: {rouge_scores['rouge2']}")
    print(f"ROUGE-L: {rouge_scores['rougeL']}")

# Sicherstellen, dass das Skript als Hauptprogramm ausgeführt wird
if __name__ == "__main__":
    main()


# In diesem Skript, das eine Demo zur Berechnung von BLEU- und ROUGE-Scores für ein Übersetzungsmodell darstellt, werden die folgenden Punkte thematisiert:
#
# 1. **Vorbereitung der Umgebung**: Installation der erforderlichen Pakete.
# 2. **Funktionen zur Berechnung**: Separate Funktionen für BLEU und ROUGE, die klar erläutern, was sie tun.
# 3. **Verwendung von Beispielen**: Konkrete Beispiele für Referenzsätze und deren Übersetzungen.
# 4. **Ausgabe der Ergebnisse**: BELU und ROUGE-Scores werden ausgegeben, um die Resultate leicht verständlich zu machen.
