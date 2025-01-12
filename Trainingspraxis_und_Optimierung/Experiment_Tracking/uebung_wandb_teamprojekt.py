"""
Aufgabenstellung:
In dieser Übung werden Sie experimentelles Tracking in einem Teamprojekt mit Weights & Biases (W&B) durchführen. 
Ihre Aufgabe ist es, ein Textklassifizierungsmodell zu trainieren und dabei die Experimente zu überwachen. 
Im Team teilen Sie sich wie folgt auf:
1. Ein Teammitglied erstellt und konfiguriert ein W&B-Projekt.
2. Ein weiteres Teammitglied implementiert ein einfaches Textklassifizierungsmodell.
3. Ein drittes Teammitglied startet das Training, loggt relevante Metriken und hyperparameter in W&B, 
   und analysiert die Ergebnisse in der W&B-Oberfläche.

Schritte:
1. Registrieren Sie sich bei Weights & Biases und erstellen Sie ein neues Projekt.
2. Installieren Sie die benötigten Python-Pakete einschließlich `wandb`.
3. Implementieren Sie den folgenden Code in Ihr Projekt.
4. Führen Sie das Training des Modells aus und verfolgen Sie die Metriken in Ihrer W&B-Projektdashboard.

Der folgende Python Code enthält die Schritte zum Einrichten von W&B, Herunterladen eines Datasets, 
Erstellen und Trainieren eines einfachen Modells zur Sentiment-Analyse.
"""

# Import necessary libraries
import wandb
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Initialize W&B project
# Initialize a new W&B run to log this experiment
wandb.login()  # Make sure to login using `wandb login` in your terminal
wandb.init(
    project="text-classification-demo",  # Replace with your project name
    config={
        "model_type": "MultinomialNB",
        "max_features": 1000,
        "alpha": 1.0,
        "test_size": 0.2,
        "random_state": 42
    }
)

# Step 2: Fetch dataset
# We use the 20 Newsgroups dataset which is a collection of text documents
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
newsgroups = fetch_20newsgroups(subset='all', categories=categories)

# Log the dataset size
wandb.config.update({"dataset_size": len(newsgroups.data)})

# Step 3: Preprocess data
# Convert text documents to TF-IDF features
tfidf = TfidfVectorizer(max_features=wandb.config.max_features)
X = tfidf.fit_transform(newsgroups.data)
y = newsgroups.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=wandb.config.test_size, random_state=wandb.config.random_state
)

# Step 4: Train model
# Initialize and train a Naive Bayes classifier
clf = MultinomialNB(alpha=wandb.config.alpha)
clf.fit(X_train, y_train)

# Step 5: Evaluate model
# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate and log accuracy
accuracy = accuracy_score(y_test, y_pred)
wandb.log({"accuracy": accuracy})

# Log detailed classification report
report = classification_report(y_test, y_pred, target_names=newsgroups.target_names, output_dict=True)
wandb.log(report)

# End the W&B run
wandb.finish()
