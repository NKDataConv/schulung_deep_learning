from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

# Constants
MODEL_ID = "bert-base-cased"
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
NUM_EPOCHS = 0.3  # Train on 30% of one epoch for faster results

# Prepare data and splits
tomatoes = load_dataset("rotten_tomatoes")
# Take smaller subset for faster training
train_data = tomatoes["train"].shuffle(seed=42).select(range(1000))
test_data = tomatoes["test"].shuffle(seed=42).select(range(200))

# Load Model and Tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Pad to the longest sequence in the batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(examples):
    """Tokenize input data"""
    return tokenizer(examples["text"], truncation=True, max_length=128)  # Added max_length for faster processing

# Tokenize train/test data
tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    """Calculate F1 score"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    load_f1 = evaluate.load("f1")
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"f1": f1}

# Training arguments for parameter tuning
training_args = TrainingArguments(
    output_dir="german_sentiment_model",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    save_strategy="no",  # Don't save checkpoints for this demo
    report_to="none",
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Trainer which executes the training process
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()
print("Training completed!")

results = trainer.evaluate()
print(f"Evaluation results: {results}")
