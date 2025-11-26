# Guía de Ejecución - Reconocimiento de Emociones en Texto (NLP)

Sigue estos pasos para activar la nueva funcionalidad.

## 1. Base de Datos (PostgreSQL)
Ejecuta el siguiente script SQL para actualizar la tabla `historial`:

```bash
psql -U postgres -d emotion_recognition -f setup_nlp.sql
```
*(Asegúrate de usar tu usuario y nombre de base de datos correctos)*

## 2. Backend Python (NLP Service)
Primero, instala las nuevas dependencias y entrena el modelo.

```bash
cd python-service
source venv/bin/activate  # O el comando para activar tu entorno virtual
pip install -r requirements.txt

# Entrenar el modelo NLP (esto generará models/nlp_model.h5)
python train_nlp.py

# Iniciar el servidor (ahora con soporte para texto)
python main.py
```
*El servidor correrá en http://0.0.0.0:8000*

## 3. Backend Java (Spring Boot)
Compila y ejecuta el servicio Java actualizado.

```bash
cd java-service
./mvnw clean install
./mvnw spring-boot:run
```
*El servidor correrá en http://localhost:8080*

## 4. Frontend Flutter
Ejecuta la aplicación móvil.

```bash
cd emotion_recognition_app
flutter run
```

## 5. Prueba Rápida (CURL)
Puedes probar el servicio Python directamente:

```bash
curl -X POST "http://localhost:8000/predict-text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Estoy muy feliz con este resultado"}'
```

O a través del servicio Java:

```bash
curl -X POST "http://localhost:8080/api/emotions/recognize-text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Tengo miedo de la oscuridad"}'
```

## Notas
- El modelo de NLP incluido en `train_nlp.py` usa un dataset sintético básico si no encuentra `data/emoevent_es.csv`. Para mejores resultados, descarga el dataset EmoEvent y colócalo en la carpeta `data`.
- La aplicación Flutter ahora tiene un botón "Analizar Texto" en la pantalla principal.
