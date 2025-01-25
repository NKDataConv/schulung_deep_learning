import pandas as pd
from keras.src.utils import to_categorical
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)

from tf_keras.models import Sequential
from tf_keras.layers import Dense

NUM_CLASSES = 2

y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)


def create_model():

    model = Sequential()
    model.add(Dense(100, input_dim=17, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = create_model()
model.summary()

from tf_keras.callbacks import TensorBoard, EarlyStopping

tensorboard_callback = TensorBoard(log_dir='logs/fit/weather')
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.fit(x_train, y_train,
          epochs=30,
          batch_size=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback, early_stopping],
          verbose=1)

beispiel_tag = x_test[:1]
vorhersage = model.predict(beispiel_tag)
print(vorhersage)
print("Vohersage Regen: ", np.round(vorhersage[0][1]*100, 2), "%")
