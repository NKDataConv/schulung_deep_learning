# Importieren der notwendigen Bibliotheken
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

# Schritt 1: Document Store erstellen
# Der Document Store speichert unsere Dokumente, die durchsucht werden sollen.
document_store = InMemoryDocumentStore()

# Schritt 2: Dokumente hinzufügen
# Fügen Sie einige Beispiel-Dokumente hinzu, die Abfragen beantworten können.
documents = [
    {
        'content': 'Python ist eine weit verbreitete Programmiersprache, die einfach zu lernen ist.',
        'meta': {'source': 'Wikimedia'}
    },
    {
        'content': 'Deep Learning ist ein Teilbereich des maschinellen Lernens.',
        'meta': {'source': 'Wikipedia'}
    },
    {
        'content': 'Haystack ist ein Framework für die Erstellung von Fragebeantwortungssystemen.',
        'meta': {'source': 'Haystack Dokumentation'}
    }
]

# Dokumente in den Document Store einfügen
document_store.write_documents(documents)

# Schritt 3: Retriever erstellen
# Ein Retriever wird verwendet, um relevante Dokumente für eine gegebene Anfrage zu finden.
retriever = EmbeddingRetriever(document_store=document_store)

# Indizieren der Dokumente
document_store.update_embeddings(retriever)

# Schritt 4: Reader erstellen
# Ein Reader analysiert die relevanten Dokumente und extrahiert die Antworten auf die Fragen.
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Schritt 5: Erstellen einer Pipeline für die Fragebeantwortung
# Die Pipeline kombiniert den Retriever und den Reader, um Antworten auf Fragen zu geben.
pipeline = ExtractiveQAPipeline(reader, retriever)

# Schritt 6: Beispielabfragen
# Hier können verschiedene Fragen gestellt werden, um Antworten zu erhalten.
questions = [
    "Was ist Python?",
    "Was ist Deep Learning?",
    "Was ist Haystack?"
]

# Antworten für jede Frage abrufen und anzeigen
for question in questions:
    print(f"Frage: {question}")
    
    # Verwenden der Pipeline, um die Antwort zu erhalten
    result = pipeline.run(query=question, params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}})
    
    # Zeigen der besten Antwort auf die Anfrage
    answer = result['answers'][0].answer
    source = result['answers'][0].meta['source']
    
    print(f"Antwort: {answer} (Quelle: {source})\n")

# Schritt 7: Abschluss
# Das System ist nun bereit, weitere Fragen zu beantworten.
print("Fragebeantwortungssystem erfolgreich eingerichtet!")

### Erläuterungen zum Skript:

# - **Document Store**: Hier wird ein InMemoryDocumentStore verwendet, um unsere Dokumente zu speichern. Es gibt auch Möglichkeiten, Dokumente in einer Datenbank wie Elasticsearch zu speichern.
#
# - **Retriever**: Der `EmbeddingRetriever` wird verwendet, um relevante Dokumente für eine Benutzeranfrage zu finden. Dieser kann auf vortrainierten Modellen basieren, um die semantische Ähnlichkeit zu verstehen.
#
# - **Reader**: Der `FARMReader` führt die eigentliche Fragebeantwortung durch. Er analysiert die gefundenen Dokumente und extrahiert spezifische Antworten.
#
# - **Pipeline**: Das zentrale Element ist die `ExtractiveQAPipeline`, die den Retriever und Reader kombiniert, um auf Benutzerfragen zu antworten.
#
# - **Fragen und Antworten**: Das Skript zeigt, wie das System verwendet werden kann, um Antworten auf mehrere vordefinierte Fragen zu erhalten.
