from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pickle
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from model_utils import preprocess_image, decode_base64_image
from config import MODEL_PATH, EMOTION_LABELS, API_HOST, API_PORT

app = FastAPI(title="Emotion Recognition API", version="1.2.0")

# Variables globales para modelos
image_model = None
nlp_model = None
tokenizer = None
nlp_label_mapping = None

# Configuración NLP
NLP_MODEL_PATH = "models/nlp_model_transformer"
MAX_LEN = 128
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def load_models():
    global image_model, nlp_model, tokenizer, nlp_label_mapping
    
    # Cargar Modelo de Imagen (TensorFlow/Keras)
    if os.path.exists(MODEL_PATH):
        try:
            image_model = keras.models.load_model(MODEL_PATH)
            print(f"Modelo de IMAGEN cargado desde {MODEL_PATH}")
        except Exception as e:
            print(f"ERROR cargando modelo de imagen: {e}")
    else:
        print(f"ADVERTENCIA: Modelo de imagen no encontrado en {MODEL_PATH}")

    # Cargar Modelo NLP (PyTorch/Transformers)
    if os.path.exists(NLP_MODEL_PATH):
        try:
            print(f"Cargando modelo NLP desde {NLP_MODEL_PATH}...")
            tokenizer = DistilBertTokenizer.from_pretrained(NLP_MODEL_PATH)
            nlp_model = DistilBertForSequenceClassification.from_pretrained(NLP_MODEL_PATH)
            nlp_model.to(DEVICE)
            nlp_model.eval()
            
            with open("models/label_encoder.pickle", 'rb') as handle:
                nlp_label_mapping = pickle.load(handle)
                
            print(f"Modelo NLP Transformer cargado exitosamente en {DEVICE}")
        except Exception as e:
            print(f"ERROR cargando modelo NLP: {e}")
    else:
        print(f"ADVERTENCIA: Modelo NLP no encontrado en {NLP_MODEL_PATH}")

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
        "message": "Emotion Recognition API (Image + Text Transformer)",
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

# Endpoint Imagen (TensorFlow)
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

# Endpoint Texto (PyTorch Transformer)
@app.post("/predict-text", response_model=PredictionResponse)
async def predict_text_emotion(request: TextRequest):
    if nlp_model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Modelo NLP no cargado")
    
    try:
        # Preprocesar texto
        encoding = tokenizer.encode_plus(
            request.text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        # Predicción
        with torch.no_grad():
            outputs = nlp_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            
        probs = probs.cpu().numpy()[0]
        predicted_idx = np.argmax(probs)
        confidence = float(probs[predicted_idx])
        
        # Mapeo inverso de etiquetas
        idx_to_label = {v: k for k, v in nlp_label_mapping.items()}
        emotion = idx_to_label[predicted_idx]
        
        probabilities = {
            idx_to_label[i]: float(probs[i])
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
