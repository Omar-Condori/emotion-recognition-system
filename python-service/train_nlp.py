import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import pickle
import os
import json

# Configuración
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 100
MODEL_PATH = "models/nlp_model.h5"
TOKENIZER_PATH = "models/tokenizer.pickle"
LABEL_ENCODER_PATH = "models/label_encoder.pickle"

# 1. Dataset (Ejemplo extendido o carga desde CSV)
# Si tienes el dataset EmoEvent, cárgalo aquí. Si no, usamos este dummy para probar.
def load_data():
    # Intentar cargar dataset real si existe
    if os.path.exists("data/emoevent_es.csv"):
        print("Cargando dataset EmoEvent...")
        df = pd.read_csv("data/emoevent_es.csv")
        return df['text'].values, df['emotion'].values
    
    print("Dataset no encontrado. Generando dataset de ejemplo para demostración...")
    # Dataset sintético básico para que el sistema funcione "out of the box"
    data = [
        ("Estoy muy feliz por este logro", "joy"),
        ("Qué alegría verte de nuevo", "joy"),
        ("Me siento genial hoy", "joy"),
        ("Es un día maravilloso", "joy"),
        ("Tengo mucho miedo de lo que pueda pasar", "fear"),
        ("Me asusta la oscuridad", "fear"),
        ("Estoy aterrorizado por las noticias", "fear"),
        ("Qué horror de película", "fear"),
        ("Estoy muy triste y deprimido", "sadness"),
        ("Me duele mucho el corazón", "sadness"),
        ("No tengo ganas de hacer nada", "sadness"),
        ("Es una pena que terminara así", "sadness"),
        ("Estoy furioso con el servicio", "anger"),
        ("Me molesta mucho tu actitud", "anger"),
        ("Odio cuando pasa esto", "anger"),
        ("Qué rabia me da", "anger"),
        ("No puedo creer lo que veo", "surprise"),
        ("Wow, esto es increíble", "surprise"),
        ("Me has dejado sin palabras", "surprise"),
        ("Qué sorpresa tan agradable", "surprise"),
        ("Esto me da asco", "disgust"),
        ("Qué comida tan desagradable", "disgust"),
        ("Me repugna esa idea", "disgust"),
        ("Huele fatal aquí", "disgust")
    ]
    # Multiplicamos los datos para tener suficiente para entrenar (solo demo)
    data = data * 20 
    df = pd.DataFrame(data, columns=['text', 'emotion'])
    return df['text'].values, df['emotion'].values

# 2. Preprocesamiento
def train():
    if not os.path.exists("models"):
        os.makedirs("models")

    texts, labels = load_data()
    
    # Tokenización
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Codificación de etiquetas
    label_mapping = {label: i for i, label in enumerate(np.unique(labels))}
    numeric_labels = np.array([label_mapping[l] for l in labels])
    categorical_labels = to_categorical(numeric_labels)
    
    # Guardar artefactos
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(LABEL_ENCODER_PATH, 'wb') as handle:
        pickle.dump(label_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Vocabulario: {len(tokenizer.word_index)}")
    print(f"Etiquetas: {label_mapping}")

    # 3. Modelo (Bi-LSTM)
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(len(label_mapping), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # 4. Entrenamiento
    print("Iniciando entrenamiento...")
    history = model.fit(padded_sequences, categorical_labels, epochs=20, batch_size=16, validation_split=0.2)
    
    model.save(MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

if __name__ == "__main__":
    train()
