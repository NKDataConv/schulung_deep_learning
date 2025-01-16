import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import time

# Constants
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
VECTOR_DIMENSION = 768  # Dimension for the chosen model

# Example texts
example_texts = [
    "Neuronale Netze sind ein Teilgebiet der künstlichen Intelligenz.",
    "Transformer Netzwerke ermöglichen die Verarbeitung von Sequenzen.",
    "Künstliche Intelligenz ist ein interdisziplinäres Forschungsgebiet.",
    "Maschinelles Lernen ermöglicht Computern, aus Erfahrungen zu lernen.",
    "Deep Learning ist eine spezielle Form des maschinellen Lernens.",
    "Reinforcement Learning basiert auf dem Prinzip von Belohnung und Bestrafung.",
    "Natural Language Processing ermöglicht die Verarbeitung natürlicher Sprache.",
    "Computer Vision befasst sich mit der visuellen Wahrnehmung durch KI.",
    "Künstliche Intelligenz findet Anwendung in der Robotik und Automation.",
    "Ethische Aspekte spielen eine wichtige Rolle in der KI-Entwicklung."
]

def create_vector_database(texts: List[str]) -> tuple:
    """
    Create a FAISS vector database from input texts.
    """

    # Load model and create embeddings
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts)

    # Create FAISS index
    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    index.add(embeddings)

    print("\nVektordatenbank erfolgreich erstellt!")
    return index, model, texts
        

def semantic_search(query: str, index, model, texts: List[str], k: int = 3) -> List[tuple]:
    """
    Perform semantic search for a query.
    """

    print(f"\nSuche nach: '{query}'")

    # Create query embedding
    query_embedding = model.encode([query])

    # Search in the index
    distances, indices = index.search(query_embedding, k)

    # Create results
    results = []
    for distance, idx in zip(distances[0], indices[0]):
        results.append((texts[idx], float(distance)))

    return results


# Create vector database
index, model, texts = create_vector_database(example_texts)

# Example search queries
queries = [
    "Was ist Deep Learning?",
    "Ethik in der KI",
    "Roboter und Automatisierung"
]

# Perform searches
for query in queries:
    results = semantic_search(query, index, model, texts)
    print(f"\nErgebnisse für '{query}':")
    for text, score in results:
        print(f"- {text} (Ähnlichkeit: {score:.3f})")
