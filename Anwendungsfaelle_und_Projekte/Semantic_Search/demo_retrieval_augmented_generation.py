from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time


# Konstanten
MODEL_NAME = "dbmdz/german-gpt2"  # Kleines deutsches GPT-2 Modell
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


class RAGSystem:
    def __init__(self):
        # Initialisierung der Modelle
        self.retriever = SentenceTransformer(EMBEDDING_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
        self.generator = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.index = None
        self.documents = []
        
    def add_documents(self, documents: List[str]):
        print("Verarbeite Dokumente...")
        self.documents = documents
        
        # Erstelle Embeddings mit Fortschrittsanzeige
        embeddings = self.retriever.encode(documents)

        # Erstelle FAISS Index
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        query_embedding = self.retriever.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[idx] for idx in indices[0]]
    
    def generate(self, query: str, retrieved_docs: List[str]) -> str:
        # Bereite Kontext vor
        context = " ".join(retrieved_docs)
        prompt = f"Kontext: {context}\n\nFrage: {query}\n\nAntwort:"
        
        # Tokenisiere Input
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, 
                               truncation=True, padding=True)
        inputs = inputs.to(DEVICE)
        
        outputs = self.generator.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extrahiere nur die Antwort nach dem "Antwort:" Teil
        answer = generated_text.split("Antwort:")[-1].strip()
        return answer


# Initialisiere RAG System
rag = RAGSystem()
rag.add_documents(example_texts)

# Beispielanfrage
query = "Was ist Deep Learning?"

# Hole relevante Dokumente
retrieved_docs = rag.retrieve(query)
print(f"\nGefundene relevante Dokumente für '{query}':")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"{i}. {doc}")

# Generiere Antwort
answer = rag.generate(query, retrieved_docs)
print(f"\nGenerierte Antwort: {answer}")
