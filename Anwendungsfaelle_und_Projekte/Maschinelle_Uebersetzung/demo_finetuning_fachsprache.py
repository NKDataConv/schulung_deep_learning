# Import notwendiger Bibliotheken
import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Schritt 1: Modell und Tokenizer initialisieren
# Wir verwenden ein vortrainiertes MarianMT-Modell für die maschinelle Übersetzung.
model_name = "Helsinki-NLP/opus-mt-de-en"  # Deutsche zu Englische Übersetzung
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Schritt 2: Daten vorbereiten
# In einem echten Szenario würden wir ein spezifisches Dataset laden, das unsere Fachsprache enthält.
# Hier verwenden wir ein Beispiel-Dataset.
dataset = load_dataset("ted_hrlr_translate", 'pt-en')
train_dataset = dataset['train']
valid_dataset = dataset['validation']

# Eine Funktion zur Tokenisierung des Textes
def preprocess_function(examples):
    # Hier werden die Texte aus der Quelle und der Zielübersetzung tokenisiert
    inputs = examples['translation']['pt']  # Ausgangssprache (z.B. Portugiesisch)
    targets = examples['translation']['en']  # Zielsprache (z.B. Englisch)
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    # Die Zieltexte tokenisieren
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Schritt 3: Daten tokenisieren
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True)

# Schritt 4: Batch-Größe und weitere Trainingsparameter setzen
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Schritt 5: Trainingsargumente definieren
training_args = TrainingArguments(
    output_dir="./results",          # Verzeichnis für die Ausgaben
    evaluation_strategy="epoch",     # Evaluierung nach jeder Epoche
    learning_rate=2e-5,              # Lernrate
    per_device_train_batch_size=16,  # Batch-Größe für das Training
    per_device_eval_batch_size=16,   # Batch-Größe für die Evaluierung
    num_train_epochs=3,              # Anzahl der Trainingsepochen
    weight_decay=0.01,               # L2 Regularisierung
)

# Schritt 6: Trainer initialisieren
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    data_collator=data_collator,
)

# Schritt 7: Training des Modells
trainer.train()

# Schritt 8: Modell speichern
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Schritt 9: Beispielübersetzung mit dem feingetunten Modell
def translate(text):
    # Text tokenisieren
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Modellaufruf
    with torch.no_grad():
        translated = model.generate(**inputs, num_beams=4, max_length=128)
    # Rückübersetzung
    return tokenizer.batch_decode(translated, skip_special_tokens=True)

# Beispieltext für die Übersetzung
example_text = "Das ist ein Beispiel für maschinelles Lernen in der Fachsprache."
translated_text = translate(example_text)

# Ausgabe der Übersetzung
print(f"Originaltext: {example_text}")
print(f"Übersetzung: {translated_text[0]}")

