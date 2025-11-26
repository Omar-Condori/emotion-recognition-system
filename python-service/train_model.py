# train_model.py - Script de entrenamiento optimizado para Mac M3
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.utils import class_weight
from config import DATA_DIR, MODEL_DIR, MODEL_PATH, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, EMOTION_LABELS

# Configuración para Apple Silicon M3
print("Configurando TensorFlow para Apple Silicon M3...")
print(f"TensorFlow version: {tf.__version__}")

# Verificar disponibilidad de Metal
physical_devices = tf.config.list_physical_devices()
print(f"Dispositivos disponibles: {physical_devices}")

# Forzar uso de GPU con Metal Performance Shaders
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detectada: {gpus}")
    else:
        print("No se detectó GPU, usando CPU")
except Exception as e:
    print(f"Error configurando GPU: {e}")

def create_cnn_model():
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(EMOTION_LABELS), activation='softmax')
    ])
    
    return model

def load_fer2013_data():
    print(f"Cargando datos desde: {DATA_DIR}")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator

def train():
    print("Iniciando entrenamiento del modelo...")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    train_gen, val_gen = load_fer2013_data()
    
    print(f"Clases detectadas: {train_gen.class_indices}")
    print(f"Total de imágenes de entrenamiento: {train_gen.samples}")
    print(f"Total de imágenes de validación: {val_gen.samples}")
    
    model = create_cnn_model()
    
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stop, reduce_lr]
    
    # Calcular pesos de clases para balancear el entrenamiento
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Pesos de clases: {class_weights_dict}")

    print(f"\nEntrenando por {EPOCHS} épocas...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    print(f"\nModelo guardado en: {MODEL_PATH}")
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"Precisión final de entrenamiento: {final_train_acc:.4f}")
    print(f"Precisión final de validación: {final_val_acc:.4f}")
    
    return history

if __name__ == "__main__":
    with tf.device('/GPU:0'):
        history = train()
    print("Entrenamiento completado exitosamente!")