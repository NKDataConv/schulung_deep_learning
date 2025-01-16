from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from datasets import load_dataset


DATASET_NAME = "germeval_14"  # German sentiment dataset
dataset = load_dataset(DATASET_NAME, trust_remote_code=True)

MODEL_ID = "..."  # German BERT model
