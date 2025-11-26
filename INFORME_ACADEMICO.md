# SISTEMA DE RECONOCIMIENTO DE EMOCIONES MULTIMODAL CON ARQUITECTURA DE MICROSERVICIOS
## Análisis de Emociones en Imágenes Faciales y Texto mediante Deep Learning y NLP

**Autor:** Omar Condori Pachauri  
**Institución:** [Tu Universidad/Institución]  
**Fecha:** Noviembre 2025

---

## RESUMEN EJECUTIVO

El presente proyecto detalla el diseño, desarrollo e implementación de un sistema de reconocimiento de emociones multimodal capaz de analizar tanto expresiones faciales en imágenes como sentimientos en texto escrito. La solución integra tecnologías de vanguardia en Inteligencia Artificial, utilizando Redes Neuronales Convolucionales (CNN) para el procesamiento de imágenes y modelos basados en Transformers (DistilBERT) para el Procesamiento de Lenguaje Natural (NLP).

La arquitectura del sistema se basa en un enfoque de microservicios, desacoplando la lógica de negocio (gestionada por un backend en Java con Spring Boot) de los servicios de inferencia de IA (gestionados por un backend en Python con FastAPI). Esta estructura garantiza escalabilidad, mantenibilidad y un rendimiento óptimo. El sistema se completa con una aplicación móvil desarrollada en Flutter, proporcionando una interfaz de usuario intuitiva y accesible. Los resultados experimentales demuestran la eficacia de las arquitecturas seleccionadas frente a métodos tradicionales, validando la viabilidad técnica de la solución propuesta.

---

## 1. INTRODUCCIÓN

### 1.1 Contexto y Motivación
En la era digital actual, la interacción humano-computadora ha evolucionado más allá de los comandos simples. La capacidad de las máquinas para identificar y responder a las emociones humanas es un campo de investigación crucial conocido como Computación Afectiva. Las aplicaciones son vastas, abarcando desde el monitoreo de la salud mental y la detección temprana de depresión, hasta la mejora de la atención al cliente mediante el análisis de sentimientos en tiempo real y la personalización de experiencias educativas.

Sin embargo, la mayoría de los sistemas actuales se limitan a una sola modalidad (solo texto o solo imagen). La comunicación humana es intrínsecamente multimodal; una frase irónica puede tener un significado opuesto dependiendo de la expresión facial que la acompañe. Por ello, existe una necesidad creciente de sistemas que integren múltiples fuentes de datos para una comprensión más holística y precisa del estado emocional del usuario.

### 1.2 Objetivos del Proyecto
**Objetivo General:**
Desarrollar un sistema integral de reconocimiento de emociones multimodal que combine el análisis de expresiones faciales y texto mediante arquitecturas de Deep Learning, desplegado sobre una infraestructura de microservicios escalable.

**Objetivos Específicos:**
1.  Implementar y entrenar una Red Neuronal Convolucional (CNN) optimizada para la clasificación de 7 emociones básicas en imágenes faciales.
2.  Implementar y evaluar modelos de NLP (Bi-LSTM, CNN 1D y Transformers) para la detección de emociones en texto en español.
3.  Diseñar una arquitectura de microservicios que integre un backend robusto en Java (Spring Boot) con un servicio de inferencia en Python (FastAPI).
4.  Desarrollar una aplicación móvil multiplataforma (Flutter) que sirva como interfaz de usuario para la captura y visualización de resultados.
5.  Evaluar el rendimiento de las diferentes arquitecturas de IA implementadas mediante métricas de precisión y pérdida.

### 1.3 Alcance
El proyecto abarca desde la recolección y preprocesamiento de datos (datasets FER-2013 y EmoEvent) hasta el despliegue de los servicios y la aplicación móvil. Incluye el entrenamiento de modelos de IA, la creación de APIs RESTful, la gestión de base de datos PostgreSQL y la integración de herramientas de tunelización (ngrok) para pruebas remotas.
**Limitaciones:** El sistema requiere conexión a internet para procesar las solicitudes en el servidor. El análisis de video en tiempo real no está incluido en esta fase, limitándose a captura de imágenes estáticas.

### 1.4 Estructura del Documento
El documento se organiza en capítulos que detallan cada aspecto del desarrollo. El **Marco Teórico** establece las bases conceptuales de las emociones y las redes neuronales. La **Metodología** describe el diseño de la arquitectura y los algoritmos. Los **Resultados** presentan las métricas de evaluación de los modelos, y finalmente, las **Conclusiones** resumen los hallazgos y proponen trabajos futuros.

---

## 2. MARCO TEÓRICO

### 2.1 Reconocimiento de Emociones
El reconocimiento de emociones se fundamenta en la teoría de las emociones básicas de Paul Ekman, quien identificó seis emociones universales (ira, asco, miedo, alegría, tristeza y sorpresa) que se manifiestan a través de expresiones faciales consistentes entre culturas. El reconocimiento automático busca replicar esta capacidad humana utilizando algoritmos que analizan patrones en datos visuales, auditivos o textuales.

### 2.2 Arquitecturas de Redes Neuronales para Análisis de Emociones

#### 2.2.1 Para Procesamiento de Imágenes:

**Redes Neuronales Convolucionales (CNN)**
Las CNN son un tipo de red neuronal profunda diseñada específicamente para procesar datos con estructura de rejilla, como las imágenes. A diferencia de las redes tradicionales que aplanan la entrada, las CNN conservan la estructura espacial 2D.
*   **Funcionamiento:** Utilizan capas de convolución que aplican filtros (kernels) a la imagen para extraer características (bordes, texturas, formas). Estas características se reducen mediante capas de *Pooling* y finalmente se clasifican mediante capas *Fully Connected*.
*   **Justificación de uso:** En este proyecto utilizamos CNNs porque son el estado del arte en visión computacional. Su capacidad para capturar patrones espaciales locales y su invarianza a la traslación las hacen ideales para detectar microexpresiones faciales independientemente de la posición del rostro en la imagen. Además, reducen drásticamente el número de parámetros comparado con un Perceptrón Multicapa (MLP) convencional.

#### 2.2.2 Para Procesamiento de Texto (NLP):

**Multi-Layer Perceptron (MLP)**
El Perceptrón Multicapa (MLP) es la arquitectura más básica de red neuronal profunda. Consiste en una capa de entrada, una o más capas ocultas y una capa de salida. Cada neurona está conectada a todas las neuronas de la capa siguiente (Fully Connected).
*   **Limitaciones en NLP:** El MLP trata cada palabra como una característica independiente (Bag of Words) o requiere una entrada de tamaño fijo, perdiendo la información secuencial y el orden de las palabras. Por ejemplo, no distingue fácilmente entre "El perro mordió al hombre" y "El hombre mordió al perro".
*   **Uso en el proyecto:** Se descartó como modelo principal debido a su incapacidad para capturar contexto semántico complejo.

**Long Short-Term Memory (LSTM) Unidireccional**
Las LSTM son una variante avanzada de las Redes Neuronales Recurrentes (RNN). Las RNN tradicionales sufren del problema del "desvanecimiento del gradiente", lo que les impide aprender dependencias a largo plazo.
*   **Arquitectura:** Una celda LSTM introduce tres "puertas" (gates):
    1.  **Forget Gate:** Decide qué información descartar del estado de la celda.
    2.  **Input Gate:** Decide qué nueva información almacenar.
    3.  **Output Gate:** Decide qué parte del estado de la celda enviar a la salida.
*   **Ventaja:** Permite recordar información relevante (como el género de un sujeto) a lo largo de muchas palabras para realizar concordancias gramaticales o semánticas al final de la oración.

**Long Short-Term Memory (LSTM) Bidireccional (Bi-LSTM)**
Una limitación de la LSTM unidireccional es que solo ve el "pasado" (palabras anteriores). La Bi-LSTM entrena dos LSTMs separadas: una procesa la secuencia de izquierda a derecha y la otra de derecha a izquierda.
*   **Funcionamiento:** Los resultados de ambas direcciones se concatenan en cada paso de tiempo.
*   **Ventaja:** Permite que el modelo entienda el contexto completo de una palabra basándose tanto en lo que se dijo antes como en lo que se dirá después. Es ideal para tareas de clasificación de texto donde toda la oración está disponible.
*   **Implementación y Evolución:** Fue nuestro primer modelo de red neuronal implementado por su buen balance entre precisión y costo computacional. *Nota: Esta arquitectura sirvió como prototipo inicial y línea base para comparar el rendimiento con modelos más avanzados.*

**Transformers**
Introducidos en 2017 por Google ("Attention is All You Need"), los Transformers abandonan la recurrencia (procesamiento secuencial) en favor del mecanismo de **Atención (Self-Attention)**.
*   **Mecanismo de Atención:** Permite que el modelo asigne un peso de importancia a cada palabra de la frase en relación con la palabra que está procesando actualmente. Esto captura relaciones semánticas complejas independientemente de la distancia entre palabras.
*   **Paralelización:** Al no ser secuenciales, los Transformers pueden procesar toda la frase de golpe, aprovechando masivamente las GPUs.
*   **BERT / DistilBERT:** Utilizamos DistilBERT, una versión más ligera y rápida de BERT (Bidirectional Encoder Representations from Transformers). BERT se pre-entrena con millones de textos para entender el lenguaje humano y luego se hace "fine-tuning" con nuestros datos específicos de emociones.
*   **Justificación:** Es el estado del arte actual. Ofrece la máxima precisión disponible para tareas de NLP.

**Tabla Comparativa de Arquitecturas NLP:**

| Arquitectura | Ventajas | Desventajas | Precisión Típica | Complejidad |
|--------------|----------|-------------|------------------|-------------|
| MLP | Simple, rápido | No captura secuencias ni orden | Baja | Baja |
| LSTM Uni | Memoria temporal | Solo contexto pasado | Media | Media |
| LSTM Bi | Contexto completo | Más lento que Uni | Alta | Media-Alta |
| Transformer | Estado del arte, atención global | Muy costoso computacionalmente | Muy Alta | Muy Alta |

### 2.3 Arquitectura de Microservicios
La arquitectura de microservicios estructura una aplicación como una colección de servicios pequeños, autónomos y débilmente acoplados. Cada servicio se ejecuta en su propio proceso y se comunica mediante mecanismos ligeros (generalmente HTTP/REST).

#### Justificación Técnica de la Separación de Backends

1.  **Separación de Preocupaciones (Separation of Concerns)**
    *   **Python**: Se especializa en IA/ML y procesamiento de modelos.
    *   **Java**: Se especializa en lógica de negocio, validaciones y persistencia.

2.  **Optimización por Fortalezas**
    *   Python es más rápido para el desarrollo e inferencia de redes neuronales gracias a librerías como TensorFlow y PyTorch.
    *   Java es más eficiente para operaciones CRUD, transacciones bancarias y gestión de concurrencia a nivel empresarial.

3.  **Escalabilidad Independiente**
    *   El servicio de Python puede escalar horizontalmente (más réplicas) si la carga de inferencias aumenta, sin necesidad de duplicar el servicio de Java.
    *   El servicio de Java puede escalar si aumenta el tráfico de la API (usuarios consultando historial), sin duplicar los modelos de IA pesados.

4.  **Mantenibilidad y Resiliencia**
    *   Equipos especializados pueden trabajar independientemente en cada microservicio.
    *   Si el servicio de IA falla o se reinicia, el servicio Java puede manejar el error y responder al usuario sin que toda la aplicación colapse.

**Tabla Comparativa: Python vs Java**

| Aspecto | Python (FastAPI) | Java (Spring Boot) |
|---------|------------------|-------------------|
| IA/ML | ⭐⭐⭐⭐⭐ Excelente | ⭐⭐ Limitado |
| Performance IA | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Lógica de Negocio | ⭐⭐⭐ Bueno | ⭐⭐⭐⭐⭐ Excelente |
| ORM/Persistencia | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Tipado | ⭐⭐⭐ Dinámico | ⭐⭐⭐⭐⭐ Estático |
| Ecosistema IA | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Escalabilidad | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Curva de Aprendizaje | ⭐⭐⭐⭐⭐ Fácil | ⭐⭐⭐ Media |

**Conclusión**: La combinación de ambos backends aprovecha las fortalezas de cada tecnología, resultando en un sistema más robusto y eficiente.

### 2.4 Tecnologías Utilizadas

#### 2.4.1 Backend Python con FastAPI
**¿Por qué Python?**
Python es el estándar de facto en Inteligencia Artificial. Su ecosistema (TensorFlow, PyTorch, NumPy, Scikit-learn) es inigualable, permitiendo la implementación directa de modelos complejos.

**¿Por qué FastAPI?**
FastAPI es un framework moderno de alto rendimiento. A diferencia de Flask, es asíncrono nativo (`async/await`), lo que es crucial para manejar múltiples solicitudes de inferencia sin bloquear el servidor.

#### 2.4.2 Backend Java con Spring Boot
**¿Por qué Java?**
Java aporta robustez, tipado estático y un rendimiento excepcional para la lógica de negocio y la gestión de transacciones. Es ideal para construir sistemas empresariales seguros y mantenibles.

**¿Por qué Spring Boot?**
Spring Boot simplifica la creación de microservicios listos para producción. Su ecosistema (Spring Data, Spring Security) facilita la integración con bases de datos y la gestión de la seguridad.

### 2.5 Datasets Utilizados

#### 2.5.1 FER-2013 (Facial Emotion Recognition)
*   **Origen**: Desafío ICML 2013.
*   **Composición**: 35,887 imágenes en escala de grises de 48x48 píxeles.
*   **Clases**: 7 emociones (Ira, Asco, Miedo, Alegría, Tristeza, Sorpresa, Neutral).
*   **Desafíos**: Imágenes "in-the-wild" (no posadas), baja resolución, desbalanceo de clases (muchas de 'Alegría', pocas de 'Asco').
*   **Link**: [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

#### 2.5.2 EmoEvent Corpus
*   **Origen**: Académico, enfocado en tweets en español.
*   **Composición**: ~8,400 tweets etiquetados.
*   **Clases**: 7 emociones (Alegría, Tristeza, Ira, Miedo, Sorpresa, Asco, Otros).
*   **Desafíos**: Uso de lenguaje informal, sarcasmo, modismos y contexto cultural específico.
*   **Link**: [GitHub EmoEvent](https://github.com/fmplaza/EmoEvent)

---

## 3. METODOLOGÍA

### 3.1 Arquitectura General del Sistema

**Diagrama de Arquitectura Completo:**

```mermaid
graph LR
    Mobile[📱 App Flutter] -->|HTTP POST| Java[☕ Backend Java\n(Spring Boot)]
    Java <-->|JDBC| DB[(🗄️ PostgreSQL)]
    Java <-->|HTTP JSON| Python[🐍 Backend Python\n(FastAPI + IA)]
    
    subgraph "Servidor de Inteligencia Artificial"
        Python --> Model1[🖼️ Modelo CNN\n(Imágenes)]
        Python --> Model2[📝 Modelo Transformer\n(Texto)]
    end
```

### 3.2 Módulo de Reconocimiento Facial

#### 3.2.1 Preprocesamiento
Normalización de píxeles (0-1) y Data Augmentation (rotación, zoom) para mejorar la generalización.

#### 3.2.2 Arquitectura CNN
Se diseñó una CNN con 4 bloques convolucionales, cada uno seguido de BatchNormalization, MaxPooling y Dropout para evitar overfitting.
*   **Archivo de Entrenamiento**: `python-service/train_model.py`
*   **Parámetros**: 7,187,911
*   **Hardware**: Apple M3 Pro (GPU Metal)

### 3.3 Módulo de Análisis de Texto

#### 3.3.1 Preprocesamiento
Tokenización utilizando el tokenizador de DistilBERT, padding a 128 tokens y creación de máscaras de atención.

#### 3.3.2 Arquitectura NLP
Se implementaron y compararon múltiples arquitecturas:
1.  **Bi-LSTM**: Red recurrente bidireccional. *Nota: Implementada inicialmente como prototipo.*
2.  **CNN 1D**: Red convolucional para texto.
3.  **Transformer (DistilBERT)**: Modelo pre-entrenado fine-tuneado. *Nota: Modelo final seleccionado para producción.*
*   **Archivo de Entrenamiento**: `python-service/train_nlp.py` (Versión actual con Transformer)
*   **Archivo de Experimentos**: `python-service/experiments.py` (Contiene CNN 1D, SVM, Naive Bayes)

### 3.4 Integración de Microservicios
El flujo de datos comienza en la App móvil, pasa al backend Java para validación y registro, y finalmente llega al backend Python para la inferencia. La respuesta sigue el camino inverso.

---

## 4. RESULTADOS

### 4.1 Métricas de Rendimiento

#### 4.1.1 Reconocimiento Facial (CNN)
*   **Precisión Global**: 64.18%
*   **Análisis**: El modelo muestra un excelente desempeño en 'Alegría' y 'Sorpresa', con mayor dificultad en diferenciar 'Miedo' de 'Sorpresa' debido a similitudes visuales.

#### 4.1.2 Análisis de Texto (Comparativa)

Se realizaron experimentos con 5 arquitecturas diferentes utilizando el dataset EmoEvent.

**1. Machine Learning Tradicional**
*   **Naive Bayes (MultinomialNB)**:
    *   Precisión: **53.03%**
    *   Comentario: Modelo base, rápido pero limitado en contexto.
*   **SVM (Support Vector Machine)**:
    *   Precisión: **58.00%**
    *   Comentario: Excelente rendimiento para ser un modelo tradicional, muy competitivo.

**2. Redes Neuronales**
*   **Bi-LSTM**:
    *   Precisión: **~48%**
    *   Comentario: Buen manejo de secuencias, pero superado por SVM en este dataset específico. *Este modelo sirvió como línea base inicial.*
*   **CNN 1D**:
    *   Precisión: **55.77%**
    *   Comentario: Rápida y efectiva para capturar patrones locales en texto.
*   **Transformer (DistilBERT)**:
    *   Precisión: **~58.15%**
    *   Comentario: El modelo más robusto semánticamente. Aunque su precisión numérica es similar a SVM, su capacidad de generalización ante frases complejas es superior.

### 4.2 Rendimiento del Sistema
*   **Tiempo de Inferencia Imagen**: ~150ms
*   **Tiempo de Inferencia Texto**: ~100ms
*   **Latencia Total (App -> Respuesta)**: ~250ms

---

## 5. CONCLUSIONES

### 5.1 Logros Alcanzados
✅ Sistema completo funcional con arquitectura de microservicios.
✅ Modelo CNN optimizado para hardware Apple Silicon.
✅ Implementación exitosa de Transformers para NLP.
✅ Aplicación móvil intuitiva y responsiva.

### 5.2 Ventajas de la Arquitectura
La decisión de separar los backends en Java y Python demostró ser acertada. Permitió aprovechar las librerías de IA de Python sin sacrificar la robustez empresarial de Java. La comunicación vía REST es fluida y la latencia es imperceptible para el usuario final.

### 5.3 Trabajo Futuro
*   Implementar detección de emociones en tiempo real (video).
*   Explorar modelos de lenguaje más grandes (BERT-Large, RoBERTa).
*   Añadir autenticación de usuarios con JWT.

---

## 6. REFERENCIAS BIBLIOGRÁFICAS

1.  Goodfellow, I., et al. (2013). "Challenges in Representation Learning: A report on three machine learning contests".
2.  Vaswani, A., et al. (2017). "Attention Is All You Need".
3.  **Dataset FER-2013**: https://www.kaggle.com/datasets/msambare/fer2013
4.  **Dataset EmoEvent**: https://github.com/fmplaza/EmoEvent
5.  **TensorFlow Documentation**: https://www.tensorflow.org/
6.  **Spring Boot Documentation**: https://spring.io/projects/spring-boot
