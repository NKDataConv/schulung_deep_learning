# Aufgabe: Anpassung des Elasticsearch Index für bessere Trefferqualität
#
# Ziel der Übung:
# Sie sollen lernen, wie man einen Elasticsearch Index konfiguriert, um die Qualität der Suchergebnisse
# für eine Fragebeantwortungsanwendung zu verbessern.
#
# Aufgabenstellung:
# 1. Erstellen Sie einen neuen Index mit einem angepassten Mapping in Elasticsearch.
# 2. Importieren Sie einige Beispieldaten in den Index.
# 3. Passen Sie die Konfiguration des Index an, um die Relevanz der Treffer bei Suchanfragen zu erhöhen.
#    Dafür sollen verschiedene Analysetechniken, wie z.B. N-Gramme oder Synonyme, eingesetzt werden.
# 4. Führen Sie einen Vergleich durch, um zu zeigen, wie sich die Anpassungen auf die Suchergebnisse auswirken.
#
# Hinweis: Sie benötigen eine laufende Elasticsearch-Instanz und das Python Elasticsearch Paket:
# `pip install elasticsearch`

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Initialisierung des Elasticsearch Clients
es = Elasticsearch("http://localhost:9200")

# Name des Index
index_name = "text_analyze_index"

# Schritt 1: Anpassung des Mappings zur Verbesserung der Trefferqualität
index_mapping = {
    "settings": {
        "analysis": {
            "filter": {
                "my_ngram_filter": {
                    "type": "ngram",
                    "min_gram": 3,
                    "max_gram": 5
                },
                "synonym_filter": {
                    "type": "synonym",
                    "synonyms": [
                        "customer, client",
                        "big, large, huge"
                    ]
                }
            },
            "analyzer": {
                "custom_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "my_ngram_filter", "synonym_filter"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "custom_analyzer"
            }
        }
    }
}

# Lösche den Index, wenn er bereits existiert
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print(f"Bestehender Index '{index_name}' gelöscht.")

# Erstelle einen neuen Index mit dem Mapping
es.indices.create(index=index_name, body=index_mapping)
print(f"Neuer Index '{index_name}' erstellt.")

# Schritt 2: Importieren von Beispieldaten
documents = [
    {"content": "The client is looking for a big data solution."},
    {"content": "Our customer prefers large datasets and analysis."},
    {"content": "Big corporations need comprehensive solutions."},
    {"content": "He is a huge supporter of open source technologies."}
]

# Indizieren der Dokumente
bulk(es, [{"_index": index_name, "_source": doc} for doc in documents])
print(f"{len(documents)} Dokumente wurden in den Index '{index_name}' importiert.")

# Schritt 3: Suchanfragen vor und nach der Indexanpassung
def search_and_print(query_text):
    response = es.search(index=index_name, body={
        "query": {
            "match": {
                "content": query_text
            }
        }
    })
    print(f"Suchergebnisse für '{query_text}':")
    for hit in response['hits']['hits']:
        print(f" - {hit['_source']['content']} (Score: {hit['_score']})")

# Suche nach dem Begriff "large" vor und nach Synonym- und N-Gram-Anwendung
search_and_print("large")

# Fazit: Durch die Einstellung des Index mit Synonymen und N-Grammen sollte die Trefferqualität
# bei verwandten Begriffen und unterschiedlichen Textfragmenten nun höher sein.
