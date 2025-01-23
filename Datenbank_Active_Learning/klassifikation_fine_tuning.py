import sqlite3
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature
import torch
import os

# Constants
MODEL_ID = "bert-base-cased"
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
NUM_EPOCHS = 1
DATABASE_PATH = "rotten_tomatoes.db"
RANDOM_SEED = 42
EXPERIMENT_NAME = "sentiment_classification"
MODEL_NAME = "sentiment_bert_model"

def load_data_from_sqlite():
    """Load review data from SQLite database"""
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            query = "SELECT text, label FROM reviews"
            df = pd.read_sql_query(query, conn)
            
            # Convert sentiment labels to binary (assuming positive/negative classification)
            # df['label'] = df['label'].map({'positive': 1, 'negative': 0})
            
            # Split into train and test sets
            train_df, test_df = train_test_split(
                df, 
                test_size=0.2, 
                random_state=RANDOM_SEED
            )
            
            # Convert to dictionary format similar to HuggingFace datasets
            train_data = {
                'text': train_df['text'].tolist(),
                'label': train_df['label'].tolist()
            }
            test_data = {
                'text': test_df['text'].tolist(),
                'label': test_df['label'].tolist()
            }
            
            return train_data, test_data
            
    except sqlite3.Error as e:
        print(f"Database error occurred: {str(e)}")
        raise
    except Exception as e:
        print(f"An error occurred while loading data: {str(e)}")
        raise

# Load data from SQLite
try:
    print("Loading data from SQLite database...")
    train_data, test_data = load_data_from_sqlite()
except Exception as e:
    print(f"Failed to load data: {str(e)}")
    raise

# Load Model and Tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Pad to the longest sequence in the batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(examples):
    """Tokenize input data"""
    # Handle both dictionary and batch inputs
    if isinstance(examples, dict):
        texts = examples["text"]
    else:
        texts = [example["text"] for example in examples]
    
    # Ensure we get all the necessary fields
    return tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors=None  # Return python lists
    )

# Create dataset-like objects
class SimpleDataset:
    def __init__(self, data_dict):
        self.data = data_dict
    
    def __len__(self):
        return len(self.data['text']) if 'text' in self.data else len(self.data['input_ids'])
    
    def __getitem__(self, idx):
        item = {}
        # For tokenized data
        if 'input_ids' in self.data:
            item['input_ids'] = torch.tensor(self.data['input_ids'][idx])
            item['attention_mask'] = torch.tensor(self.data['attention_mask'][idx])
            if 'token_type_ids' in self.data:
                item['token_type_ids'] = torch.tensor(self.data['token_type_ids'][idx])
        # For raw data
        elif 'text' in self.data:
            item['text'] = self.data['text'][idx]
        
        item['labels'] = torch.tensor(self.data['label'][idx])
        return item
    
    def map(self, func, batched=True):
        if batched:
            processed = func(self.data)
            # Add labels to processed data
            processed['labels'] = self.data['label']
            # Convert all lists to numpy arrays
            processed = {k: np.array(v) for k, v in processed.items()}
        else:
            processed = {
                'input_ids': [],
                'attention_mask': [],
                'labels': self.data['label']
            }
            for i in range(len(self)):
                item_processed = func({'text': [self.data['text'][i]]})
                processed['input_ids'].append(item_processed['input_ids'][0])
                processed['attention_mask'].append(item_processed['attention_mask'][0])
            processed = {k: np.array(v) for k, v in processed.items()}
        
        return SimpleDataset(processed)

# Convert to dataset objects
train_dataset = SimpleDataset(train_data)
test_dataset = SimpleDataset(test_data)

# Tokenize train/test data
print("Tokenizing training data...")
tokenized_train = train_dataset.map(preprocess_function, batched=True)
print("Tokenizing test data...")
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# After loading data
print("Sample of train_data:", {k: v[:2] for k, v in train_data.items()})
print("Sample of tokenized_train:", {k: v[:2] for k, v in tokenized_train.data.items()})

# After tokenization
print("\nVerifying data structure:")
print("Keys in first training item:", tokenized_train[0].keys())
print("Shape of input_ids:", len(tokenized_train[0]['input_ids']))
print("Shape of attention_mask:", len(tokenized_train[0]['attention_mask']))
print("Label type:", type(tokenized_train[0]['labels']))
print("Label value:", tokenized_train[0]['labels'])

# Before creating trainer
print("Verifying tokenized dataset structure...")
print("Train dataset length:", len(tokenized_train))
print("Test dataset length:", len(tokenized_test))

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

# Save the model with MLflow
try:
    # Set up MLflow tracking to use local directory
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"bert_finetuning_{RANDOM_SEED}") as run:
        # Log parameters
        mlflow.log_params({
            "model_name": MODEL_ID,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "weight_decay": training_args.weight_decay,
        })
        
        # Log metrics
        mlflow.log_metrics({
            "f1_score": results["eval_f1"],
            "eval_loss": results["eval_loss"]
        })
        
        # Save model artifacts
        mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": tokenizer
            },
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
        
        print(f"Model saved in MLflow with run ID: {run.info.run_id}")
        print(f"Model registered as: {MODEL_NAME}")
        print(f"Model artifacts stored in: {os.path.abspath('./mlruns')}")
        
except Exception as e:
    print(f"Error saving model to MLflow: {str(e)}")
    raise
