import sqlite3
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from sklearn.model_selection import train_test_split
import torch
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "bert-base-cased"
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
NUM_EPOCHS = 1
DATABASE_PATH = "rotten_tomatoes.db"
RANDOM_SEED = 42
MODEL_DIR = "sentiment_model"

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data_from_sqlite():
    """Load review data from SQLite database"""
    try:
        if not Path(DATABASE_PATH).exists():
            raise FileNotFoundError(f"Database file {DATABASE_PATH} not found!")
            
        with sqlite3.connect(DATABASE_PATH) as conn:
            query = "SELECT text, label FROM reviews"
            df = pd.read_sql_query(query, conn)
            
            if len(df) == 0:
                raise ValueError("No data found in the reviews table!")
            
            # Add debug information
            logger.info(f"Loaded {len(df)} reviews from database")
            logger.info(f"DataFrame columns: {df.columns}")
            logger.info(f"Sample of labels: {df['label'].head()}")
            
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=RANDOM_SEED
            )
            
            train_data = {
                'text': train_df['text'].tolist(),
                'labels': train_df['label'].tolist()
            }
            test_data = {
                'text': test_df['text'].tolist(),
                'labels': test_df['label'].tolist()
            }
            
            # Verify data format
            logger.info(f"Train data keys: {train_data.keys()}")
            logger.info(f"Test data keys: {test_data.keys()}")
            logger.info(f"Sample train labels: {train_data['labels'][:5]}")
            
            return train_data, test_data
            
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

class SimpleDataset:
    def __init__(self, data_dict):
        # Add validation and debugging
        if not isinstance(data_dict, dict):
            raise ValueError(f"Expected dict, got {type(data_dict)}")
        
        required_keys = ['text', 'labels'] if 'text' in data_dict else ['input_ids', 'attention_mask', 'labels']
        missing_keys = [key for key in required_keys if key not in data_dict]
        if missing_keys:
            raise KeyError(f"Missing required keys: {missing_keys}. Available keys: {list(data_dict.keys())}")
            
        self.data = data_dict
        logger.info(f"Dataset initialized with keys: {list(self.data.keys())}")
        logger.info(f"Number of samples: {len(self)}")
    
    def __len__(self):
        return len(self.data['text']) if 'text' in self.data else len(self.data['input_ids'])
    
    def __getitem__(self, idx):
        try:
            item = {}
            if 'input_ids' in self.data:
                item['input_ids'] = torch.tensor(self.data['input_ids'][idx])
                item['attention_mask'] = torch.tensor(self.data['attention_mask'][idx])
                if 'token_type_ids' in self.data:
                    item['token_type_ids'] = torch.tensor(self.data['token_type_ids'][idx])
            elif 'text' in self.data:
                item['text'] = self.data['text'][idx]
            
            # Ensure labels exist and are properly formatted
            if 'labels' not in self.data:
                raise KeyError(f"'labels' not found in data. Available keys: {list(self.data.keys())}")
            
            item['labels'] = torch.tensor(self.data['labels'][idx], dtype=torch.long)
            return item
        except Exception as e:
            logger.error(f"Error in __getitem__ at idx {idx}: {str(e)}")
            raise
    
    def map(self, func, batched=True):
        try:
            if batched:
                processed = func(self.data)
            else:
                processed = {
                    'input_ids': [],
                    'attention_mask': [],
                    'labels': self.data['labels']
                }
                for i in range(len(self)):
                    item_processed = func({'text': [self.data['text'][i]]})
                    processed['input_ids'].append(item_processed['input_ids'][0])
                    processed['attention_mask'].append(item_processed['attention_mask'][0])
            
            # Verify processed data has required keys
            required_keys = ['input_ids', 'attention_mask', 'labels']
            missing_keys = [key for key in required_keys if key not in processed]
            if missing_keys:
                raise KeyError(f"Processing resulted in missing keys: {missing_keys}")
            
            processed = {k: np.array(v) for k, v in processed.items()}
            return SimpleDataset(processed)
        except Exception as e:
            logger.error(f"Error in map function: {str(e)}")
            raise

def compute_metrics(eval_pred):
    """Calculate F1 score"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    load_f1 = evaluate.load("f1")
    return {"f1": load_f1.compute(predictions=predictions, references=labels)["f1"]}

def save_model(model, tokenizer):
    """Save model and tokenizer to disk"""
    try:
        logger.info(f"Saving model to {MODEL_DIR}")
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Load data
        train_data, test_data = load_data_from_sqlite()
        logger.info("Data loaded successfully")
        
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # Create datasets
        train_dataset = SimpleDataset(train_data)
        logger.info("Train dataset created successfully")
        test_dataset = SimpleDataset(test_data)
        logger.info("Test dataset created successfully")
        
        # Tokenize datasets
        logger.info("Starting tokenization...")
        tokenized_train = train_dataset.map(lambda x: tokenizer(
            x["text"], truncation=True, max_length=128, padding='max_length'
        ))
        logger.info("Train dataset tokenized successfully")
        tokenized_test = test_dataset.map(lambda x: tokenizer(
            x["text"], truncation=True, max_length=128, padding='max_length'
        ))
        logger.info("Test dataset tokenized successfully")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="temp_trainer",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=0.01,
            save_strategy="no",
            report_to="none",
            evaluation_strategy="epoch"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            compute_metrics=compute_metrics,
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed!")
        
        # Evaluate
        results = trainer.evaluate()
        logger.info(f"Evaluation results: {results}")
        
        # Save model
        save_model(model, tokenizer)
        
    except Exception as e:
        logger.error(f"Training process failed: {str(e)}")
        raise
