"""
Aufgabenstellung:
In dieser Übung richten Sie ein einfaches Monitoring-System ein, das grundlegende Metriken eines Python-basierten Webservers überwacht, 
und visualisieren diese mit Prometheus und Grafana. 
Folgen Sie diesen Schritten:

1. Installieren Sie die erforderlichen Pakete: Flask, prometheus_client, und requests.
2. Implementieren Sie einen einfachen Flask-Webserver mit mindestens zwei Endpunkten: '/hello' und '/metrics'.
3. Der Endpunkt '/hello' gibt eine Willkommensnachricht zurück. 
4. Der Endpunkt '/metrics' dient Prometheus zur Überwachung. Verwenden Sie die prometheus_client Bibliothek, um Metriken zu sammeln, 
   wie z.B. Anfragenanzahl und Antwortzeiten.
5. Starten Sie den Flask-Server.
6. Richten Sie einen Prometheus-Server ein, um die Metriken von Ihrem Flask-Server zu sammeln.
7. Visualisieren Sie die gesammelten Metriken mit Grafana.
8. Dokumentieren Sie jeden Schritt sorgfältig und achten Sie darauf, dass das Python Skript gut kommentiert ist.

Hinweis: Diese Aufgabe erfordert grundlegende Kenntnisse in Flask, Prometheus und Grafana. Stellen Sie sicher, dass Prometheus und Grafana 
  auf Ihrem System installiert sind.

Jetzt das Python Skript, das einen einfachen Flask-Server mit prometheus_client einrichtet.
"""

from flask import Flask
from prometheus_client import start_http_server, Summary
from prometheus_client.core import CollectorRegistry
from prometheus_client import Counter, Histogram, generate_latest

import time
import random

app = Flask(__name__)

# Zähler für die Anzahl der Anfragen
REQUEST_COUNT = Counter('hello_request_count', 'Total number of hello requests')
# Histogramm zur Messung der Antwortzeiten
REQUEST_LATENCY = Histogram('hello_request_latency_seconds', 'Latency of hello requests in seconds')

@app.route('/hello')
def hello():
    """Einfacher Endpunkt, der eine Willkommensnachricht ausgibt."""
    # Zähle die Anfrage
    REQUEST_COUNT.inc()
    
    # Start timer for latency
    start_time = time.time()
    
    # Simulate variable processing time
    processing_time = random.uniform(0.1, 0.5)
    time.sleep(processing_time)
    
    # Beobachte die bearbeitungszeit
    REQUEST_LATENCY.observe(time.time() - start_time)

    return "Willkommen zu unserer Beispiel-Flask-Anwendung!"

@app.route('/metrics')
def metrics():
    """Bereitstellung der Metriken für Prometheus."""
    # Rückgabe der gesammelten Metriken als Text
    return generate_latest()

if __name__ == '__main__':
    # Starten Sie den HTTP-Server und lauschen Sie auf Port 8000
    start_http_server(8000)
    # Starten Sie die Flask-Anwendung auf Port 8080, wählen Sie den host='0.0.0.0' um Verbindung von außen zu ermöglichen
    app.run(host='0.0.0.0', port=8080)
