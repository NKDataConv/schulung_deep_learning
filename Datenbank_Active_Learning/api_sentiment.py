from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging
import os
from tqdm import tqdm
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = "sentiment_model"
HOST = "127.0.0.1"
PORT = 8000

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment classification using fine-tuned BERT model",
    version="1.0.0"
)

class TextInput(BaseModel):
    text: str

class ModelLoader:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model from disk"""
        try:
            if not os.path.exists(MODEL_DIR):
                raise FileNotFoundError(f"Model directory {MODEL_DIR} not found!")
            
            logger.info("Loading model...")
            for _ in tqdm(range(5), desc="Loading model"):
                time.sleep(0.1)
            
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            logger.info("Model loaded successfully!")
            return {"model": self.model, "tokenizer": self.tokenizer}
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

# Initialize model loader
model_loader = ModelLoader()

@app.on_event("startup")
async def startup_event():
    """Load model during startup"""
    try:
        app.state.model = model_loader.load_model()
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.post("/predict", response_model=Dict[str, float])
async def predict_sentiment(input_data: TextInput):
    """Predict sentiment for input text"""
    if not hasattr(app.state, 'model'):
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    try:
        # Get model and tokenizer
        model = app.state.model["model"]
        tokenizer = app.state.model["tokenizer"]
        
        # Prepare input
        inputs = tokenizer(
            input_data.text,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Get prediction
        outputs = model(**inputs)
        probabilities = outputs.logits.softmax(dim=-1).tolist()[0]
        
        return {
            "negative": float(probabilities[0]),
            "positive": float(probabilities[1])
        }
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": hasattr(app.state, 'model')}

if __name__ == "__main__":
    uvicorn.run("api_sentiment:app", host=HOST, port=PORT, reload=True) 