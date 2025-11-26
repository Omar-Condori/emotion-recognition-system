# INFORME TÉCNICO: SISTEMA DE RECONOCIMIENTO DE EMOCIONES (ACTUALIZADO)

## Capítulo 8: PROCESO DE ENTRENAMIENTO DEL MODELO

### 8.1. Dataset FER-2013 (Imágenes)
*   **Fuente**: Facial Expression Recognition 2013 (Kaggle)
*   **Total imágenes de entrenamiento**: 28,709
*   **Validación**: 7,178
*   **Resolución**: 48x48 (escala de grises)
*   **Clases**: 7 emociones (angry, disgust, fear, happy, neutral, sad, surprise)

### 8.2. Distribución por clase (Imágenes)
*   angry: 4,953
*   disgust: 547
*   fear: 5,121
*   happy: 8,989
*   neutral: 6,198
*   sad: 6,077
*   surprise: 4,002

### 8.3. Preprocesamiento Imágenes (ImageDataGenerator)
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)
```

### 8.4. Hiperparámetros (Imágenes)
*   **Batch Size**: 64
*   **Épocas**: 50
*   **Learning Rate**: 0.001
*   **Optimizador**: Adam
*   **Loss**: Categorical Crossentropy

### 8.5. Callbacks
```python
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
```

### 8.6. Resultados del Entrenamiento (Imágenes)
*   **Mejor modelo**: Época 16 (aprox)
*   **Precisión de Entrenamiento**: ~63.84%
*   **Precisión de Validación**: ~64.18%

---

### 8.7. Dataset de Texto (NLP) - **[NUEVO]**
*   **Fuente**: EmoEvent (Tweets en español) / Dataset Sintético de respaldo.
*   **Clases**: 6 emociones (joy, sadness, anger, fear, surprise, disgust).
*   **Formato**: Texto libre en español.

### 8.8. Preprocesamiento de Texto - **[NUEVO]**
*   **Tokenización**: Vocabulario limitado a 10,000 palabras más frecuentes.
*   **Padding**: Secuencias ajustadas a longitud fija de 100 tokens (`post-padding`).
*   **Encoding**: Transformación de etiquetas a vectores categóricos (`One-Hot Encoding`).

### 8.9. Arquitectura Modelo NLP (Bi-LSTM) - **[NUEVO]**
*   **Embedding**: Dimensión 100.
*   **Capas Recurrentes**: 
    *   `Bidirectional(LSTM(64))`
    *   `Bidirectional(LSTM(32))`
*   **Regularización**: `Dropout(0.5)`
*   **Salida**: `Dense(softmax)`

### 8.10. Resultados Entrenamiento NLP - **[NUEVO]**
*   **Estrategia**: Se aplicaron **Class Weights** para compensar el desbalanceo del dataset (ej. 'others' vs 'disgust').
*   **Precisión Entrenamiento**: ~97.9% (Epoch 20)
*   **Precisión Validación**: ~46.4% (Epoch 20)
*   **Pérdida Entrenamiento**: 0.04
*   **Pérdida Validación**: 3.92

---

## Capítulo 9: CONFIGURACIÓN Y DESPLIEGUE

### 9.1. Configuración del Entorno Python
```bash
# Crear entorno virtual
python3.11 -m venv venv
source venv/bin/activate

# Instalar dependencias (Actualizado con NLP)
pip install --upgrade pip
pip install tensorflow-macos tensorflow-metal
pip install fastapi uvicorn pillow numpy opencv-python
pip install python-multipart pydantic scikit-learn matplotlib
# Dependencias nuevas para Texto:
pip install pandas nltk
```

### 9.2. Configuración del Entorno Java
```bash
# Verificar Java 17
java -version

# Compilar proyecto
cd java-service
mvn clean install

# Ejecutar
mvn spring-boot:run
```

### 9.3. Configuración de PostgreSQL
```sql
CREATE DATABASE emotion_recognition;
CREATE USER emotion_user WITH PASSWORD 'emotion_pass';
GRANT ALL PRIVILEGES ON DATABASE emotion_recognition TO emotion_user;
```

### 9.4. Configuración de Flutter
```bash
# Instalar dependencias
flutter pub get

# Ejecutar en macOS
flutter run -d macos
# Generar APK para Android
flutter build apk --release
```

---

## Capítulo 10: PRUEBAS Y RESULTADOS

### 10.1. Pruebas de Endpoints

**Python (IA)**
```bash
# Health check
curl http://localhost:8000/health

# Predicción Imagen
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"image":"BASE64"}'

# Predicción Texto [NUEVO]
curl -X POST http://localhost:8000/predict-text -H "Content-Type: application/json" -d '{"text":"Estoy muy feliz hoy"}'
```

**Java (Backend)**
```bash
# Health check
curl http://localhost:8080/api/emotions/health

# Reconocimiento Imagen
curl -X POST http://localhost:8080/api/emotions/recognize -H "Content-Type: application/json" -d '{"image":"BASE64"}'

# Reconocimiento Texto [NUEVO]
curl -X POST http://localhost:8080/api/emotions/recognize-text -H "Content-Type: application/json" -d '{"text":"Tengo miedo del futuro"}'
```

### 10.2. Prueba de Integración Completa
*   Backend Python iniciado (puerto 8000).
*   Backend Java iniciado (puerto 8080).
*   PostgreSQL corriendo (puerto 5432).
*   **ngrok** exponiendo puerto 8080 (HTTPS).
*   App Flutter ejecutándose en Android/iOS.
*   **Flujo Imagen**: Captura -> Java -> Python (CNN) -> BD -> App.
*   **Flujo Texto**: Input -> Java -> Python (LSTM) -> BD -> App.

### 10.3. Métricas de Rendimiento (Estimado)
*   Tiempo inferencia Imagen: ~100-150 ms
*   Tiempo inferencia Texto: ~20-50 ms **[Más rápido]**
*   Tiempo comunicación Java↔Python: ~50-70 ms
*   Tamaño modelo Imagen: ~27 MB
*   Tamaño modelo Texto: ~4 MB

---

## Capítulo 11: CONCLUSIONES

### 11.1. Logros Alcanzados
*   Arquitectura microservicios implementada correctamente.
*   Integración exitosa entre componentes (Java, Python, Flutter).
*   **Sistema Multimodal**: Capacidad de reconocer emociones tanto en rostros como en texto.
*   API REST completa y persistencia con PostgreSQL.
*   Despliegue accesible remotamente vía ngrok.

### 11.2. Posibles Mejoras Futuras
*   Aumentar precisión con transfer learning (VGG16, ResNet para imágenes; BERT para texto).
*   Implementar data augmentation avanzado.
*   Autenticación JWT.
*   Dockerizar servicios.

---

## Apéndice A: Scripts y comandos útiles

### A.1. SQL: Creación de tabla (Actualizado)
```sql
CREATE TABLE historial (
    id BIGSERIAL PRIMARY KEY,
    emocion VARCHAR(50) NOT NULL,
    confianza DOUBLE PRECISION NOT NULL,
    fecha_deteccion TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    tiempo_procesamiento BIGINT,
    probabilidades TEXT,
    tipo VARCHAR(20) DEFAULT 'IMAGE' -- Nueva columna para diferenciar TEXT vs IMAGE
);
-- Crear índice para búsquedas rápidas por tipo
CREATE INDEX idx_historial_tipo ON historial(tipo);
```
