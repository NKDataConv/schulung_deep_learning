# Importieren der notwendigen Bibliotheken
from fastapi import FastAPI, HTTPException, UploadFile, File
import mlflow.pyfunc
import uvicorn
import numpy as np
from PIL import Image
import io

mlflow.set_tracking_uri('http://127.0.0.1:5000')

# Major variables
MODEL_NAME = "MNISTModel"

# Schritt 1: Modell laden
# Laden des neuesten Modells aus dem MLflow-Registry
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")
print(f"Successfully loaded model: {MODEL_NAME}")

# Schritt 2: Erstellen einer FastAPI-Anwendung
app = FastAPI(title="MNIST Digit Predictor",
             description="Upload an image of a handwritten digit (28x28 pixels) to get a prediction")


def process_image(file: UploadFile) -> np.ndarray:
    """
    Process uploaded image file to match MNIST format
    """
    try:
        # Read image file
        contents = file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype='float32')
        image_array = image_array / 255.0
        
        return image_array
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


# Schritt 4: Erstellen einer Endpoint-Funktion
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """
    Upload an image file of a handwritten digit.
    The image will be preprocessed to match MNIST format (28x28 grayscale).
    Returns the predicted digit.
    """
    try:
        # Schritt 4a: Vorverarbeitung des Bildes
        image_array = process_image(file)
        
        # Reshape für das Modell (flatten zu 784 Dimensionen)
        processed_image = image_array.reshape(1, 28 * 28)

        # Schritt 4b: Modellvorhersage
        prediction = model.predict(processed_image)
        prediction = np.argmax(prediction, axis=1)
        prediction = prediction.tolist()

        # Schritt 4c: Rückgabe der Vorhersage
        return {
            "filename": file.filename,
            "prediction": prediction[0],
            "content_type": file.content_type
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Schritt 5: Starten der FastAPI-Anwendung
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)