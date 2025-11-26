import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
import os
import pickle

# Configuración
MODEL_NAME = 'distilbert-base-multilingual-cased'
MODEL_PATH = "models/nlp_model_transformer"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    if os.path.exists("data/emoevent_es.csv"):
        print("Cargando dataset EmoEvent...")
        df = pd.read_csv("data/emoevent_es.csv", sep='\t')
        return df['tweet'].values, df['emotion'].values
    else:
        raise FileNotFoundError("No se encontró data/emoevent_es.csv")

def train():
    if not os.path.exists("models"):
        os.makedirs("models")

    print(f"Usando dispositivo: {DEVICE}")

    texts, labels = load_data()
    
    # Mapeo de etiquetas
    unique_labels = np.unique(labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = [label_map[l] for l in labels]
    
    # Guardar mapeo
    with open("models/label_encoder.pickle", 'wb') as f:
        pickle.dump(label_map, f)
        
    print(f"Etiquetas: {label_map}")

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(texts, numeric_labels, test_size=0.2, random_state=42, stratify=numeric_labels)

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    # Datasets
    train_dataset = EmotionDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = EmotionDataset(X_val, y_val, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Class Weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"Pesos: {class_weights}")

    # Modelo
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(unique_labels))
    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Entrenamiento
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss # Hugging Face models calculate loss automatically if labels are provided, but let's use our weighted loss
            
            # Recalculate loss with weights
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss}")
        
        # Validación
        model.eval()
        val_accuracy = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                val_accuracy += (predictions == labels).sum().item()
        
        avg_val_acc = val_accuracy / len(X_val)
        print(f"Val Accuracy: {avg_val_acc}")

    print("Guardando modelo...")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print("Modelo guardado exitosamente.")

if __name__ == "__main__":
    train()
