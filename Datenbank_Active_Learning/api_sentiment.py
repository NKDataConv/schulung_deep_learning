import mlflow
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from typing import Dict, List, Optional
import sys
from tqdm import tqdm
import time
import sqlite3
import asyncio
from datetime import datetime
import logging

# Constants
MLFLOW_TRACKING_URI = "file:./mlruns"
MODEL_NAME = "sentiment_bert_model"
HOST = "127.0.0.1"
PORT = 8000
DATABASE_PATH = "rotten_tomatoes.db"
MODEL_RELOAD_INTERVAL = 60  #  in seconds

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment classification using fine-tuned BERT model",
    version="1.0.0"
)

# Pydantic models for request validation
class TextInput(BaseModel):
    text: str

class ReviewInput(BaseModel):
    text: str
    label: int  # 0 for negative, 1 for positive

class ReviewBatchInput(BaseModel):
    reviews: List[ReviewInput]

class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_uri = f"models:/{MODEL_NAME}/latest"
        
    def load_model(self):
        """Load the latest version of the model from MLflow"""
        try:
            print("Loading model from MLflow...")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            # Check if model exists in registry
            try:
                model_details = mlflow.tracking.MlflowClient().get_registered_model(MODEL_NAME)
                print(f"Found model {MODEL_NAME} in registry")
            except mlflow.exceptions.MlflowException as e:
                if "RESOURCE_DOES_NOT_EXIST" in str(e):
                    raise RuntimeError(f"Model {MODEL_NAME} not found in MLflow registry. Please ensure the model is registered.")
                raise e

            # Show loading animation
            for _ in tqdm(range(10), desc="Loading model"):
                time.sleep(0.1)
                
            # Get the latest version of the model
            model = mlflow.transformers.load_model(model_uri=self.model_uri)
            print("Model loaded successfully!")
            return model
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

class DatabaseResponse(BaseModel):
    message: str
    inserted_count: int
    error: Optional[str] = None

# Initialize model loader
model_loader = ModelLoader()

# Add new class for model reloading
class ModelReloader:
    def __init__(self):
        self.last_reload_time = None
        self.is_reloading = False
        
    async def reload_model_periodically(self, app: FastAPI):
        """Periodically reload the model in the background"""
        while True:
            try:
                if not self.is_reloading:
                    self.is_reloading = True
                    print(f"Reloading model at {datetime.now()}")
                    
                    # Load new model
                    new_model = model_loader.load_model()
                    
                    # Update app state with new model
                    app.state.model = new_model
                    
                    self.last_reload_time = datetime.now()
                    print(f"Model successfully reloaded at {self.last_reload_time}")
                    
                    self.is_reloading = False
                
                # Wait for next reload interval
                await asyncio.sleep(MODEL_RELOAD_INTERVAL)
                
            except Exception as e:
                error_msg = f"Error during model reload: {str(e)}"
                print(error_msg)
                logging.error(error_msg)
                self.is_reloading = False
                # Wait a shorter time before retry in case of error
                await asyncio.sleep(60)

# Initialize model reloader
model_reloader = ModelReloader()

# Modify startup event to start the periodic reload task
@app.on_event("startup")
async def startup_event():
    """Load model during startup and start periodic reload"""
    try:
        # Initial model load
        app.state.model = model_loader.load_model()
        model_reloader.last_reload_time = datetime.now()
        
        # Start background task for periodic reload
        background_tasks = BackgroundTasks()
        background_tasks.add_task(model_reloader.reload_model_periodically, app)
        
        # Create task for periodic reload
        asyncio.create_task(model_reloader.reload_model_periodically(app))
        
    except RuntimeError as e:
        error_msg = f"Startup failed: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        sys.exit(1)
    except Exception as e:
        error_msg = f"Unexpected error during startup: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        sys.exit(1)

@app.post("/predict", response_model=Dict[str, float])
async def predict_sentiment(input_data: TextInput):
    """
    Predict sentiment for input text
    Returns probability scores for positive and negative sentiments
    """
    if not hasattr(app.state, 'model') or app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    try:
        # Get model from app state
        model = app.state.model
        
        # Tokenize input text
        tokenizer = model["tokenizer"]
        classifier = model["model"]
        
        # Prepare input for model
        inputs = tokenizer(
            input_data.text,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Get model prediction
        outputs = classifier(**inputs)
        probabilities = outputs.logits.softmax(dim=-1).tolist()[0]
        
        # Return prediction probabilities
        return {
            "negative": float(probabilities[0]),
            "positive": float(probabilities[1])
        }
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logging.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

class DatabaseHandler:
    @staticmethod
    def insert_reviews(reviews: List[ReviewInput]) -> DatabaseResponse:
        """Insert reviews into the SQLite database"""
        try:
            with sqlite3.connect(DATABASE_PATH, timeout=10) as conn:
                cursor = conn.cursor()
                
                # Insert reviews
                cursor.executemany(
                    "INSERT INTO reviews (text, label) VALUES (?, ?)",
                    [(review.text, review.label) for review in reviews]
                )
                
                inserted_count = cursor.rowcount
                conn.commit()
                
                return DatabaseResponse(
                    message="Reviews inserted successfully",
                    inserted_count=inserted_count
                )
                
        except sqlite3.Error as e:
            error_msg = f"Database error: {str(e)}"
            print(error_msg)
            return DatabaseResponse(
                message="Failed to insert reviews",
                inserted_count=0,
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(error_msg)
            return DatabaseResponse(
                message="Failed to insert reviews",
                inserted_count=0,
                error=error_msg
            )

# Initialize database handler
db_handler = DatabaseHandler()

@app.post("/reviews/add", response_model=DatabaseResponse)
async def add_review(review: ReviewInput):
    """
    Add a single review to the database
    """
    return db_handler.insert_reviews([review])

@app.post("/reviews/batch", response_model=DatabaseResponse)
async def add_reviews_batch(reviews: ReviewBatchInput):
    """
    Add multiple reviews to the database in a batch
    """
    if not reviews.reviews:
        raise HTTPException(
            status_code=400,
            detail="No reviews provided in the batch"
        )
    
    # Validate labels
    invalid_labels = [r for r in reviews.reviews if r.label not in [0, 1]]
    if invalid_labels:
        raise HTTPException(
            status_code=400,
            detail="Invalid labels found. Labels must be 0 (negative) or 1 (positive)"
        )
    
    return db_handler.insert_reviews(reviews.reviews)

@app.get("/reviews/stats")
async def get_database_stats():
    """
    Get statistics about the reviews in the database
    """
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM reviews")
            total_count = cursor.fetchone()[0]
            
            # Get label distribution
            cursor.execute("""
                SELECT label, COUNT(*) as count 
                FROM reviews 
                GROUP BY label
            """)
            label_counts = dict(cursor.fetchall())
            
            return {
                "total_reviews": total_count,
                "label_distribution": {
                    "negative": label_counts.get(0, 0),
                    "positive": label_counts.get(1, 0)
                }
            }
            
    except sqlite3.Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )

# Add new endpoint to check model status
@app.get("/model/status")
async def get_model_status():
    """Get information about the current model status"""
    return {
        "last_reload_time": model_reloader.last_reload_time.isoformat() 
            if model_reloader.last_reload_time else None,
        "is_reloading": model_reloader.is_reloading,
        "reload_interval_hours": MODEL_RELOAD_INTERVAL / 3600
    }

# Add new endpoint to trigger manual reload
@app.post("/model/reload")
async def trigger_model_reload():
    """Manually trigger model reload"""
    try:
        if model_reloader.is_reloading:
            raise HTTPException(
                status_code=409,
                detail="Model is currently being reloaded"
            )
        
        model_reloader.is_reloading = True
        new_model = model_loader.load_model()
        app.state.model = new_model
        model_reloader.last_reload_time = datetime.now()
        model_reloader.is_reloading = False
        
        return {
            "message": "Model reloaded successfully",
            "reload_time": model_reloader.last_reload_time.isoformat()
        }
        
    except Exception as e:
        model_reloader.is_reloading = False
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("api_sentiment:app", host=HOST, port=PORT, reload=True) 