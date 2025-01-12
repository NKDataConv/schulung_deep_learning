# Python Script: Einfache CI/CD-Pipeline mit GitHub Actions

# Dieses Skript dient zur Demonstration, wie eine einfache CI/CD-Pipeline
# mit GitHub Actions umgesetzt werden kann. Die Pipeline überprüft und
# testet unseren Python-Code automatisch, wenn Änderungen an unserem
# Repository vorgenommen werden.

# Importiere notwendige Bibliotheken
import unittest

# Beispiel-Funktion, die getestet werden soll
def add(a, b):
    """Fügt zwei Zahlen zusammen."""
    return a + b

def subtract(a, b):
    """Subtrahiert b von a."""
    return a - b

# Testfälle für die Funktionen
class TestMathFunctions(unittest.TestCase):

    def test_add(self):
        """Testet die add-Funktion."""
        self.assertEqual(add(1, 2), 3)  # Erwartet 3
        self.assertEqual(add(-1, 1), 0)  # Erwartet 0
        self.assertEqual(add(0, 0), 0)    # Erwartet 0

    def test_subtract(self):
        """Testet die subtract-Funktion."""
        self.assertEqual(subtract(5, 2), 3)  # Erwartet 3
        self.assertEqual(subtract(5, 5), 0)  # Erwartet 0
        self.assertEqual(subtract(0, 5), -5)  # Erwartet -5

# Hauptfunktion zum Ausführen der Tests
if __name__ == '__main__':
    unittest.main()

# Um die GitHub Actions zu aktivieren, erstelle die folgende .yml Datei
# in einem neuen Verzeichnis `.github/workflows` in deinem Projekt-Repository:

# .github/workflows/python-app.yml

"""
name: Python application

on: [push]

jobs:
  build:

    runs-on: ubuntu-latest

    steps:
    - name: Check out the code
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.x'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run tests
      run: |
        python -m unittest discover
"""

# Wichtige Punkte zu CI/CD:
# 1. Continuous Integration (CI): Automatisches Testen und Zusammenführen von Codeänderungen.
# 2. Continuous Deployment (CD): Automatisches Bereitstellen der Anwendung nach erfolgreichen Tests.
# 3. GitHub Actions: Ein CI/CD-Dienst von GitHub, der Workflows zur Automatisierung von Aufgaben
#    wie Testen und Bereitstellen ermöglicht.
# 4. Tests sollten immer geschrieben werden, bevor Codeänderungen in das Haupt-Repository
#    integriert werden. Dies sichert die Qualität des Codes.
