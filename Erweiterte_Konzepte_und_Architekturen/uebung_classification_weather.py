import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# Daten einlesen
df = pd.read_csv("daten/weatherAUS.csv")

# Reduzieren der Daten auf Analysevariablen
df = df.drop(['Date', 'Location', "WindDir9am", "WindDir3pm", "WindGustDir"], axis=1)

# Umgang mit NA: Löschen
df = df.dropna()

# Label Encoder, um No Yes zu 0, 1 umzuwandeln
le_encoder = LabelEncoder()
le_encoder.fit(df["RainTomorrow"])
df["RainTomorrow"] = le_encoder.transform(df["RainTomorrow"])
df["RainToday"] = le_encoder.transform(df["RainToday"])

# One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# Ziel Variable y und Features x
y = df["RainTomorrow"]
x = df.drop(["RainTomorrow", "RISK_MM"], axis=1)

#Aufteilen der Daten in Trainings, Validierungs und Testdaten
x_train, x_text, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)

