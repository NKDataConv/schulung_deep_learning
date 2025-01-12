# Aufgabe: Übung: Feintuning von BERT auf kleinem NER-Datensatz
# Ziel dieser Übung ist es, das vortrainierte BERT-Sprachmodell (z.B. 'bert-base-cased') auf einem
# Named Entity Recognition (NER) Datensatz weiter zu trainieren (Feintuning).
# Wir verwenden dazu den Hugging Face Transformers und Datasets Bibliotheken.
# Schritt 1: Installiere die notwendigen Bibliotheken, falls noch nicht geschehen:
# ```
# pip install transformers datasets
# ```
# Schritt 2: Lade einen kleinen NER-Datensatz. In dieser Übung verwenden wir den 'conll2003' Datensatz
# von Hugging Face datasets.
# Schritt 3: Bereite die Daten für das Modell vor, indem du Tokenization und Encoding der Labels vornimmst.
# Schritt 4: Konfiguriere eine Datenpipeline für das Training.
# Schritt 5: Initialisiere das BERT-Modell für die Tokenklassifikation und das dazugehörige Training.
# Schritt 6: Training durchführen.
# Schritt 7: Evaluierung des Modells auf dem Testdatensatz.

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np

# Step 1: Lade den 'bert-base-cased' Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Step 2: Lade den conll2003 Datensatz
dataset = load_dataset("conll2003")

# Eine Übersicht der Datennutzungen und Tokenizing zu schaffen
# Beispielhafte Verschlüsselung eines Textes aus dem Datensatz
example_text = dataset['train'][0]['tokens']
print("Example Text:", example_text)
tokenized_example = tokenizer(example_text, is_split_into_words=True)
print("Tokenized Example:", tokenized_example)

# Eine Funktion, die die Token-Etiketten zu Indizes konvertiert 
# benutzt werden für Token-Klassifizierung Aufgaben
def encode_labels(examples):
    # Ein simplifizierter Ansatz um die Labels zu encoden
    labels = examples["ner_tags"]
    return {"labels": labels}

# Den Dataset mit tokenisierten Eingaben versehen
# und die NER-Labels dazu encoden
dataset = dataset.map(lambda examples: tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, padding=True), batched=True)
dataset = dataset.map(encode_labels, batched=True)

# Labels zu einer Liste von Listen konvertieren
label_list = dataset['train'].features['ner_tags'].feature.names

# Step 3: Initialisiere das BERT Modell für die Tokenklassifikation
model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list))

# Step 4: Setup der Trainings- und Evaluierungsparameter
# Es ist ratsam, hier die Hardware und die gewünschten Parameter anzupassen
training_args = TrainingArguments(
    output_dir="./results",          # Verzeichnis für die Ergebnissicherung
    evaluation_strategy="epoch",     # Evaluierung nach jeder Epoche
    save_strategy="epoch",           # Modell nach jeder Epoche sichern
    learning_rate=2e-5,              # Lernrate konfigurieren
    per_device_train_batch_size=16,  # Batchgröße pro Gerät
    per_device_eval_batch_size=64,   # Batchgröße für Evaluierung
    num_train_epochs=3,              # Anzahl der Trainingsiterationen (Epoche)
    weight_decay=0.01,               # Regularisierungsparameter
)

# Step 6: Initiere das Trainer-Objekt, das für das Training und die Evaluierung verwendet wird
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    # A simple metrics function to calculate the evaluation metrics
    compute_metrics=lambda p: compute_metrics(p, label_list),
)

# Step 5 & 7: Starte das Training und Evaluierung 
trainer.train()

# Helper function to compute target metrics for NER tasks
def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    metric = load_metric("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
