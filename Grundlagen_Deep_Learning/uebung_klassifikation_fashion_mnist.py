import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import datasets

# 1. Fashion-MNIST-Daten laden
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
