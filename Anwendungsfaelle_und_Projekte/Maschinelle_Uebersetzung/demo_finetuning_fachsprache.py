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

# Konstanten definieren
MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"
DATASET_NAME = "Helsinki-NLP/opus_books"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
OUTPUT_DIR = "./results"
MODEL_SAVE_DIR = "./fine_tuned_model"

# Schritt 2: Daten vorbereiten
try:
    dataset = load_dataset(DATASET_NAME, 'de-en')
    # Teile den Trainingsdatensatz in Training und Validierung auf
    split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    valid_dataset = split_dataset['test']  # Verwende den 'test' Split als Validierungsdatensatz
    
    print(f"Trainingsdaten: {len(train_dataset)} Beispiele")
    print(f"Validierungsdaten: {len(valid_dataset)} Beispiele")
except Exception as e:
    print(f"Fehler beim Laden des Datasets: {str(e)}")
    raise

# Eine Funktion zur Tokenisierung des Textes
def preprocess_function(examples):
    try:
        # Extrahiere die Übersetzungen direkt aus dem Dataset
        if isinstance(examples['translation'], list):
            # Wenn translation eine Liste ist
            inputs = []
            targets = []
            for item in examples['translation']:
                if isinstance(item, dict) and 'de' in item and 'en' in item:
                    inputs.append(item['de'])
                    targets.append(item['en'])
        else:
            # Wenn translation ein einzelnes Dictionary ist
            inputs = examples['translation']['de']
            targets = examples['translation']['en']
        
        # Tokenisiere die Eingabetexte
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors=None  # Wichtig: Keine Tensoren zurückgeben
        )
        
        # Die Zieltexte tokenisieren
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=MAX_LENGTH,
                truncation=True,
                padding=True,
                return_tensors=None  # Wichtig: Keine Tensoren zurückgeben
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    except Exception as e:
        print(f"Fehler bei der Vorverarbeitung: {str(e)}")
        print(f"Typ von translation: {type(examples['translation'])}")
        print(f"Beispiel-Eingabe: {examples['translation']}")
        raise

# Schritt 3: Daten tokenisieren
print("Tokenisiere Trainingsdaten...")
tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenisiere Trainingsdaten"
)

print("Tokenisiere Validierungsdaten...")
tokenized_valid_dataset = valid_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=valid_dataset.column_names,
    desc="Tokenisiere Validierungsdaten"
)

# Schritt 4: Batch-Größe und weitere Trainingsparameter setzen
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Schritt 5: Trainingsargumente definieren
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    save_total_limit=2,  # Nur die letzten 2 Checkpoints behalten
    load_best_model_at_end=True,  # Bestes Modell am Ende laden
    metric_for_best_model="eval_loss",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=100,
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
model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)

# Schritt 9: Beispielübersetzung mit dem feingetunten Modell
def translate(text):
    try:
        # Text tokenisieren
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        # Modellaufruf
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                num_beams=4,
                max_length=MAX_LENGTH,
                early_stopping=True
            )
        # Rückübersetzung
        return tokenizer.batch_decode(translated, skip_special_tokens=True)
    except Exception as e:
        print(f"Fehler bei der Übersetzung: {str(e)}")
        return None

# Beispieltexte für die Übersetzung
example_texts = [
    "Das ist ein Beispiel für maschinelles Lernen in der Fachsprache.",
    "Der Autor schrieb ein faszinierendes Buch über künstliche Intelligenz.",
    "Die Geschichte spielt in einer fernen Zukunft."
]

print("\nÜbersetzungsbeispiele:")
print("-" * 50)
for text in example_texts:
    translation = translate(text)
    if translation:
        print(f"Original: {text}")
        print(f"Übersetzung: {translation[0]}")
        print("-" * 50)

