from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pickle
from model_utils import preprocess_image, decode_base64_image
from config import MODEL_PATH, EMOTION_LABELS, API_HOST, API_PORT

app = FastAPI(title="Emotion Recognition API", version="1.1.0")

# Variables globales para modelos
image_model = None
nlp_model = None
tokenizer = None
nlp_label_mapping = None

# Configuración NLP
NLP_MODEL_PATH = "models/nlp_model.h5"
TOKENIZER_PATH = "models/tokenizer.pickle"
LABEL_ENCODER_PATH = "models/label_encoder.pickle"
MAX_LEN = 100

def load_models():
    global image_model, nlp_model, tokenizer, nlp_label_mapping
    
    # Cargar Modelo de Imagen
    if os.path.exists(MODEL_PATH):
        image_model = keras.models.load_model(MODEL_PATH)
        print(f"Modelo de IMAGEN cargado desde {MODEL_PATH}")
    else:
        print(f"ADVERTENCIA: Modelo de imagen no encontrado en {MODEL_PATH}")

    # Cargar Modelo NLP
    if os.path.exists(NLP_MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        nlp_model = keras.models.load_model(NLP_MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open(LABEL_ENCODER_PATH, 'rb') as handle:
            nlp_label_mapping = pickle.load(handle)
        print(f"Modelo NLP cargado desde {NLP_MODEL_PATH}")
    else:
        print(f"ADVERTENCIA: Modelo NLP no encontrado. Ejecuta train_nlp.py")

@app.on_event("startup")
async def startup_event():
    load_models()
    print(f"API iniciada en http://{API_HOST}:{API_PORT}")

# DTOs
class ImageRequest(BaseModel):
    image: str

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    probabilities: dict

@app.get("/")
async def root():
    return {
        "message": "Emotion Recognition API (Image + Text)",
        "status": "running",
        "image_model": image_model is not None,
        "nlp_model": nlp_model is not None
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "image_model": image_model is not None,
        "nlp_model": nlp_model is not None
    }

# Endpoint Imagen (Existente)
@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(request: ImageRequest):
    if image_model is None:
        raise HTTPException(status_code=500, detail="Modelo de imagen no cargado")
    
    try:
        image_bytes = decode_base64_image(request.image)
        processed_image = preprocess_image(image_bytes)
        predictions = image_model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        emotion = EMOTION_LABELS[predicted_class]
        
        probabilities = {
            EMOTION_LABELS[i]: float(predictions[0][i])
            for i in range(len(EMOTION_LABELS))
        }
        
        return PredictionResponse(
            emotion=emotion,
            confidence=confidence,
            probabilities=probabilities
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")

# Nuevo Endpoint Texto
@app.post("/predict-text", response_model=PredictionResponse)
async def predict_text_emotion(request: TextRequest):
    if nlp_model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Modelo NLP no cargado")
    
    try:
        # Preprocesar texto
        sequences = tokenizer.texts_to_sequences([request.text])
        padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Predicción
        predictions = nlp_model.predict(padded, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        
        # Mapeo inverso de etiquetas
        idx_to_label = {v: k for k, v in nlp_label_mapping.items()}
        emotion = idx_to_label[predicted_idx]
        
        probabilities = {
            idx_to_label[i]: float(predictions[0][i])
            for i in range(len(idx_to_label))
        }
        
        return PredictionResponse(
            emotion=emotion,
            confidence=confidence,
            probabilities=probabilities
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando texto: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
