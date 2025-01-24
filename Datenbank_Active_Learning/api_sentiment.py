from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import uvicorn
from typing import Dict, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging
import os
from tqdm import tqdm
import time
import asyncio
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = "../sentiment_model"
HOST = "127.0.0.1"
PORT = 8000
RELOAD_INTERVAL = 60  # seconds
LAST_RELOAD_TIME = None

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment classification using fine-tuned BERT model",
    version="1.0.0"
)

class TextInput(BaseModel):
    text: str

class Review(BaseModel):
    text: str
    label: int

class ReviewBatch(BaseModel):
    reviews: List[Review]

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

async def periodic_model_reload():
    """Periodically reload the model"""
    global LAST_RELOAD_TIME
    while True:
        try:
            current_time = datetime.now()
            
            # Check if it's time to reload
            if (LAST_RELOAD_TIME is None or 
                (current_time - LAST_RELOAD_TIME).total_seconds() >= RELOAD_INTERVAL):
                
                logger.info("Performing periodic model reload...")
                app.state.model = model_loader.load_model()
                LAST_RELOAD_TIME = current_time
                logger.info("Model reloaded successfully")
            
            await asyncio.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in periodic model reload: {str(e)}")
            await asyncio.sleep(5)  # Wait before retrying

@app.on_event("startup")
async def startup_event():
    """Load model and start periodic reload during startup"""
    try:
        # Initial model load
        app.state.model = model_loader.load_model()
        global LAST_RELOAD_TIME
        LAST_RELOAD_TIME = datetime.now()
        
        # Start periodic reload task
        asyncio.create_task(periodic_model_reload())
        
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

@app.post("/load_data")
async def load_new_data(batch: ReviewBatch):
    """Load new review data into the database"""
    try:
        import sqlite3
        
        # Connect to database
        conn = sqlite3.connect("rotten_tomatoes.db", check_same_thread=False)
        cursor = conn.cursor()
        
        # Insert data - access Review object attributes directly
        cursor.executemany(
            "INSERT INTO reviews (text, label) VALUES (?, ?)",
            [(review.text, review.label) for review in batch.reviews]
        )
        
        # Commit and close
        conn.commit()
        conn.close()
        
        return {"message": f"Successfully loaded {len(batch.reviews)} reviews"}
        
    except Exception as e:
        error_msg = f"Failed to load data: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/model_status")
async def model_status():
    """Get model status and last reload time"""
    global LAST_RELOAD_TIME
    return {
        "model_loaded": hasattr(app.state, 'model'),
        "last_reload": LAST_RELOAD_TIME.isoformat() if LAST_RELOAD_TIME else None,
        "reload_interval_seconds": RELOAD_INTERVAL
    }

if __name__ == "__main__":
    uvicorn.run("api_sentiment:app", host=HOST, port=PORT, reload=True) 