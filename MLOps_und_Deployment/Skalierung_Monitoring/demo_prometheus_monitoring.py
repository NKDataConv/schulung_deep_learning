# Importiere erforderliche Bibliotheken
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Summary, Histogram
import random
import time

# Initialisiere das Flask-App
app = Flask(__name__)

# Erstelle eine Histogramm-Metrik für die Latenzüberwachung
REQUEST_TIME = Histogram('request_latency_seconds', 'Latenz der Anfragen in Sekunden',
                         ['method', 'endpoint'])

# Erstelle einen Summary für die insgesamt verstrichene Zeit zur Überwachung
REQUEST_DURATION = Summary('request_processing_seconds', 'Dauer der Anfrageverarbeitung in Sekunden',
                            ['method', 'endpoint'])

# Beispielendpoint zur Verarbeitung von Textdaten
@app.route('/analyze', methods=['POST'])
def analyze_text():
    # Simuliere eine zufällige Latenz zur Veranschaulichung
    time.sleep(random.uniform(0.5, 2.0))
    
    # Lese den eingehenden Text aus der Anfrage
    data = request.json
    text = data.get('text', '')

    # Hier könnte die Textanalyse oder -verarbeitung erfolgen
    # Zum Beispiel: einfache Wortzählung
    word_count = len(text.split())

    # Gebe das Ergebnis zurück
    return jsonify({"word_count": word_count})

# Dekorator zur Metriküberwachung für den jeweiligen Endpoint
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    # Berechne die verstrichene Zeit
    duration = time.time() - request.start_time
    # Aktualisiere die Histogramm-Metriken
    REQUEST_TIME.labels(method=request.method, endpoint=request.path).observe(duration)
    # Aktualisiere die Summary-Metriken
    REQUEST_DURATION.labels(method=request.method, endpoint=request.path).observe(duration)
    return response

# Hauptfunktion, um den Prometheus-Server zu starten und die Flask-App auszuführen
if __name__ == '__main__':
    # Starte den Prometheus-Metriks-Server auf Port 9091
    start_http_server(9091)
    
    # Starte die Flask-App auf Port 5000
    app.run(host='0.0.0.0', port=5000)


### Erklärungen des Codes:
# 1. **Bibliotheken Importieren**:
#    - Flask wird verwendet, um eine Webanwendung zu erstellen.
#    - Prometheus Client Bibliothek wird verwendet, um Metriken für die Latenzüberwachung zu definieren.
#
# 2. **Histogramm und Summary**:
#    - `REQUEST_TIME`: Speichert die Latenz für jede Anfrage, kategorisiert nach HTTP-Methode und Endpoint.
#    - `REQUEST_DURATION`: Gibt die gesamte Verarbeitungszeit der Anfrage als Zusammenfassung an.
#
# 3. **Flask-Endpunkt**:
#    - Der Endpoint `/analyze` simuliert die Textanalyse. In der Demo wird nur eine einfache Wortzählung durchgeführt. In einer echten Anwendung könnte hier ein Deep Learning Modell zur Textanalyse integriert werden.
#
# 4. **Latenzüberwachung**:
#    - Vor und nach der Anfrage wird die Zeit gemessen, um die Latenz in den Metriken zu erfassen.
#
# 5. **Serverstart**:
#    - Der Prometheus Metrikserver wird auf Port 9091 gestartet, sodass Prometheus darauf zugreifen kann.
#    - Die Flask-App wird auf Port 5000 gestartet, um Anfragen entgegenzunehmen.

### Nutzung:
# - Um die Demo zu starten, speichere den Code in einer Datei (z. B. `app.py`), installiere die erforderlichen Pakete (`Flask` und `prometheus_client`) und führe das Skript aus.
# - Du kannst dann mit einem HTTP-Client (z. B. Postman oder curl) an den Endpoint `/analyze` Anfragen senden, um die Latenz zu überwachen. Stelle sicher, dass Prometheus entsprechend konfiguriert ist, um die Metriken zu sammeln.