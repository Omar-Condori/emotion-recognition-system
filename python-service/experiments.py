import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

# Configuración
DATA_PATH = "data/emoevent_es.csv"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def load_data():
    if os.path.exists(DATA_PATH):
        print("Cargando dataset...")
        df = pd.read_csv(DATA_PATH, sep='\t')
        return df['tweet'].values, df['emotion'].values
    else:
        raise FileNotFoundError(f"No se encontró {DATA_PATH}")

# --- 1. Machine Learning Tradicional ---

def run_naive_bayes(X_train, X_test, y_train, y_test):
    print("\n--- Entrenando Naive Bayes (ML Tradicional 1) ---")
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Accuracy Naive Bayes: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, zero_division=0))

def run_svm(X_train, X_test, y_train, y_test):
    print("\n--- Entrenando SVM (ML Tradicional 2) ---")
    model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', class_weight='balanced'))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Accuracy SVM: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, zero_division=0))

# --- 2. Red Neuronal (CNN 1D) ---

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes, num_filters):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x) # [batch, sent_len, embed_dim]
        x = x.permute(0, 2, 1) # [batch, embed_dim, sent_len]
        
        conved = [torch.relu(conv(x)) for conv in self.convs] # [batch, num_filters, sent_len - kernel_size + 1]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] # [batch, num_filters]
        
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        return self.fc(cat)

class CNNDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item].split()
        indices = [self.vocab.get(w, self.vocab["<UNK>"]) for w in text]
        if len(indices) < self.max_len:
            indices += [self.vocab["<PAD>"]] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[item], dtype=torch.long)

def run_cnn(X_train, X_test, y_train, y_test, label_map):
    print("\n--- Entrenando CNN 1D (Red Neuronal 3) ---")
    
    # Vocabulario simple
    word_counts = {}
    for text in X_train:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1
            
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in word_counts.items():
        if count > 1:
            vocab[word] = len(vocab)
            
    # Mapeo de etiquetas
    y_train_idx = [label_map[l] for l in y_train]
    y_test_idx = [label_map[l] for l in y_test]

    train_dataset = CNNDataset(X_train, y_train_idx, vocab, 100)
    test_dataset = CNNDataset(X_test, y_test_idx, vocab, 100)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    model = TextCNN(len(vocab), 100, len(label_map), [3, 4, 5], 100).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    # Evaluar
    model.eval()
    all_preds = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(DEVICE)
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            
    print(f"Accuracy CNN: {accuracy_score(y_test_idx, all_preds):.4f}")
    print(classification_report(y_test_idx, all_preds, zero_division=0))

# --- Main ---

if __name__ == "__main__":
    texts, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    
    unique_labels = np.unique(labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    run_naive_bayes(X_train, X_test, y_train, y_test)
    run_svm(X_train, X_test, y_train, y_test)
    run_cnn(X_train, X_test, y_train, y_test, label_map)
