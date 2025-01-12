# Importieren der benötigten Bibliotheken
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Setze die Zufallszahl für Reproduzierbarkeit
tf.random.set_seed(42)

# ----------------------------
# Grundlagen für die Transformer-Implementierung
# ----------------------------

# Definiere eine Funktion für die Multi-Head Attention
class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads

        # Q, K, V Dense Layer
        self.wq = layers.Dense(d_model)  # Query
        self.wk = layers.Dense(d_model)  # Key
        self.wv = layers.Dense(d_model)  # Value

        # Dense Layer für die Ausgabewerte
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # Aufteilen der Eingabe in mehrere Köpfe
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, depth)

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # Berechnung von Q, K und V
        query = self.wq(q)  # (batch_size, seq_len, d_model)
        key = self.wk(k)    # (batch_size, seq_len, d_model)
        value = self.wv(v)  # (batch_size, seq_len, d_model)

        # Aufteilen in mehrere Köpfe
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len, depth)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Berechnungen der Attention Scores
        score = tf.matmul(query, key, transpose_b=True)  # (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            score += (mask * -1e9)  # Maskierung für Padding

        attention_weights = tf.nn.softmax(score, axis=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # Mehrdimensionale Werte durch die Gewichtung anpassen
        output = tf.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len, depth)

        # Anpassen der Dimensionen zurück auf die originale Form
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, depth)
        output = tf.reshape(output, (batch_size, -1, self.d_model))  # (batch_size, seq_len, d_model)

        return self.dense(output)  # (batch_size, seq_len, d_model)


# ----------------------------
# Encoder Layer
# ----------------------------

class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads, d_model)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),  # Feed Forward Network
            layers.Dense(d_model)  # Rückkehr zur Eingabedimension
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)  # Dropout für Regularisierung
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)  # Multi-Head Attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual Connection + Layer Norm

        ffn_output = self.ffn(out1)  # Feed Forward Network
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Residual Connection + Layer Norm


# ----------------------------
# Transformer Modell erstellen
# ----------------------------

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_position_embeddings, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.dropout = layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        # seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # Eingabewörter in Vektoren umwandeln
        x *= tf.math.sqrt(tf.cast(self.embedding.input_dim, tf.float32))  # Skalierung der Embeddings
        x = self.dropout(x, training=training)

        for i in range(len(self.encoder)):
            x = self.encoder[i](x, training=training, mask=mask)

        return x  # Ausgabewerte des Transformers


# ----------------------------
# Hyperparameter
# ----------------------------

num_layers = 2
d_model = 128
num_heads = 4
dff = 512
input_vocab_size = 10000  # Beispiel für Vokabulargröße
max_position_embeddings = 100  # Maximal erlaubte Positionen
dropout_rate = 0.1

# Modell instanziieren
transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, max_position_embeddings, dropout_rate)

# define the input shape
sample_input = tf.random.uniform((64, 38))  # Batch size 64 und Sequence length 38
output = transformer(sample_input, training=False, mask=None)  # Modellaufruf
print(output.shape)  # Ausgabeform sollte (64, 38, 128) sein
