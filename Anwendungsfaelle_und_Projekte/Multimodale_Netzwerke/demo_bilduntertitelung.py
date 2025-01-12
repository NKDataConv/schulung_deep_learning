# Importieren der notwendigen Bibliotheken
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from transformers import BertTokenizer, TFBertModel

# Funktion zum Laden und Vorverarbeiten eines Bildes
def preprocess_image(image_path):
    # Bild laden und auf die Größe (299, 299) bringen (Für InceptionV3)
    img = load_img(image_path, target_size=(299, 299))
    # Bild in ein numpy Array konvertieren
    img_array = img_to_array(img)
    # Bild für das InceptionV3 Modell vorbereiten (Normalisierung)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    return img_array

# Funktion um vortrainiertes CNN-Modell zu laden und Feature-Extraktion durchzuführen
def extract_image_features(image_path):
    # Laden des vortrainierten InceptionV3 Modells ohne die letzten Schichten
    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    
    # Bild vorverarbeiten
    img_array = preprocess_image(image_path)
    
    # Feature-Extraktion
    features = model.predict(img_array)
    return features

# Funktion zum Laden des tokenizer und des BERT-Modells
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    return tokenizer, bert_model

# Funktion zur Generierung einer Bildunterschrift
def generate_caption(features, tokenizer, bert_model):
    # Beispielhafte kurze Beschreibung für Kategorisierung
    caption_input = "a photo of"
    
    # Tokenisierung des Textes
    input_ids = tokenizer.encode(caption_input, return_tensors='tf')
    
    # Durchführen der Vorwärtsausbreitung durch BERT
    outputs = bert_model(input_ids)
    last_hidden_state = outputs.last_hidden_state
    
    # Dummy-Ganzzahl-Logik für die Caption-Gewichtung (in einer echten Anwendung würde hier das Netzwerk trainiert werden)
    scores = tf.reduce_mean(last_hidden_state, axis=1)
    
    # Zurückgeben des simulierten Scores für die künftige Caption (in der Realität sollte hier ein Decodieren stattfinden)
    return f"Caption based on features: {scores.numpy()}"

# Hauptfunktion
def main(image_path):
    # Bildmerkmale extrahieren
    features = extract_image_features(image_path)
    
    # BERT Modell und Tokenizer laden
    tokenizer, bert_model = load_bert_model()
    
    # Generiere Bildunterschrift
    caption = generate_caption(features, tokenizer, bert_model)
    
    # Ergebnisse anzeigen
    plt.imshow(load_img(image_path))  # Bild anzeigen
    plt.axis('off')  # Achsen ausblenden
    plt.title(caption)  # Titel setzen
    plt.show()  # Bild und Caption anzeigen

# Skript ausführen
if __name__ == "__main__":
    # Pfad zum Bild mit Bilduntertitelung
    image_path = 'path/to/your/image.jpg'  # Hier den richtigen Pfad zum Bild angeben
    main(image_path)

### Erläuterungen zu verschiedenen Teilen des Codes:
# 1. **Importierung**: Wir importieren die notwendigen Bibliotheken für Bildverarbeitung und NLP. TensorFlow für das Deep Learning und Hugging Face's Transformers für das BERT-Modell.
#
# 2. **Bildvorverarbeitung**: Die Funktion `preprocess_image` lädt ein Bild, bringt es auf die passende Größe und bereitet es für das CNN-Modell vor.
#
# 3. **Feature-Extraktion**: In `extract_image_features` laden wir ein vortrainiertes CNN (InceptionV3) und extrahieren die Merkmale des Bildes.
#
# 4. **BERT-Modell**: Die Funktionen `load_bert_model` laden den BERT-Tokenizierer und das Modell.
#
# 5. **Generierung von Untertiteln**: In `generate_caption` wird eine einfache Logik verwendet, um eine "Caption" basierend auf den extrahierten Bildfunktionen und BERT zu generieren.
#
# 6. **Visualisierung und Ausgabe**: Schließlich zeigt das Skript das Bild zusammen mit der generierten Bildunterschrift an.
#
# ### Hinweis:
# Dieses Skript ist eine sehr vereinfachte Darstellung des Prozesses zur Bilduntertitelung und nutzt Dummy-Logik. In der Realität würde die Caption-Generierung umfassendere Methoden wie ein trainiertes Sequence-to-Sequence-Modell erfordern.