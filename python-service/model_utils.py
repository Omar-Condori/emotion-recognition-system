import cv2
import numpy as np
from PIL import Image
import io
import base64
from config import IMG_SIZE

def preprocess_image(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # If faces are found, crop the largest one
    if len(faces) > 0:
        # Find the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Add a small margin if possible
        margin = int(w * 0.1)
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(img.shape[1], x + w + margin)
        y_end = min(img.shape[0], y + h + margin)
        
        img = img[y_start:y_end, x_start:x_end]
    
    # Resize to model input size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Reshape and normalize
    img_array = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

def decode_base64_image(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_bytes = base64.b64decode(base64_string)
    return image_bytes