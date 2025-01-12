# Übung: Einbinden von vortrainierten Embeddings in ein Klassifikationsmodell

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

data = load_dataset("rotten_tomatoes")

# Select a stratified subset of the train and test data
train_data, _, train_labels, _ = train_test_split(
    data["train"]["text"], data["train"]["label"], 
    test_size=0.8, stratify=data["train"]["label"], random_state=42
)

test_data, _, test_labels, _ = train_test_split(
    data["test"]["text"], data["test"]["label"], 
    test_size=0.8, stratify=data["test"]["label"], random_state=42
)

# Encode the selected subset
train_embeddings = model.encode(train_data)
test_embeddings = model.encode(test_data)

print("Train embeddings shape:", train_embeddings.shape)

clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, train_labels)

y_pred = clf.predict(test_embeddings)
accuracy = (y_pred == test_labels).mean()
print("Accuracy:", accuracy)