package com.emotion.recognition.service;

import com.emotion.recognition.dto.EmotionRequest;
import com.emotion.recognition.dto.EmotionResponse;
import com.emotion.recognition.dto.PythonPredictionResponse;
import com.emotion.recognition.entity.Historial;
import com.emotion.recognition.repository.HistorialRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class EmotionService {

    private final RestTemplate restTemplate;
    private final HistorialRepository historialRepository;
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${python.service.url}")
    private String pythonServiceUrl;

    public EmotionResponse processEmotion(EmotionRequest request) {
        long startTime = System.currentTimeMillis();

        try {
            log.info("Enviando imagen al servicio Python...");
            PythonPredictionResponse prediction = callPythonService(request.getImage());
            long processingTime = System.currentTimeMillis() - startTime;

            log.info("Predicción recibida: {}", prediction.getEmotion());
            Historial historial = saveToDatabase(prediction, processingTime, "IMAGE");

            return EmotionResponse.builder()
                    .id(historial.getId())
                    .emotion(prediction.getEmotion())
                    .confidence(prediction.getConfidence())
                    .timestamp(historial.getFechaDeteccion())
                    .processingTime(processingTime)
                    .probabilities(prediction.getProbabilities())
                    .build();

        } catch (Exception e) {
            log.error("Error procesando emoción: {}", e.getMessage(), e);
            throw new RuntimeException("Error al procesar la imagen: " + e.getMessage());
        }
    }

    private PythonPredictionResponse callPythonService(String base64Image) {
        String url = pythonServiceUrl + "/predict";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        Map<String, String> requestBody = new HashMap<>();
        requestBody.put("image", base64Image);

        HttpEntity<Map<String, String>> entity = new HttpEntity<>(requestBody, headers);

        try {
            ResponseEntity<PythonPredictionResponse> response = restTemplate.exchange(
                    url,
                    HttpMethod.POST,
                    entity,
                    PythonPredictionResponse.class);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            } else {
                throw new RuntimeException("Respuesta inválida del servicio Python");
            }
        } catch (Exception e) {
            log.error("Error llamando al servicio Python: {}", e.getMessage());
            throw new RuntimeException("No se pudo conectar con el servicio de predicción: " + e.getMessage());
        }
    }

    public List<Historial> getHistorial() {
        return historialRepository.findTop10ByOrderByFechaDeteccionDesc();
    }

    public List<Historial> getHistorialByEmocion(String emocion) {
        return historialRepository.findByEmocion(emocion);
    }

    public EmotionResponse processTextEmotion(com.emotion.recognition.dto.TextEmotionRequest request) {
        long startTime = System.currentTimeMillis();

        try {
            log.info("Enviando texto al servicio Python...");
            PythonPredictionResponse prediction = callPythonTextService(request.getText());
            long processingTime = System.currentTimeMillis() - startTime;

            log.info("Predicción de texto recibida: {}", prediction.getEmotion());
            Historial historial = saveToDatabase(prediction, processingTime, "TEXT");

            return EmotionResponse.builder()
                    .id(historial.getId())
                    .emotion(prediction.getEmotion())
                    .confidence(prediction.getConfidence())
                    .timestamp(historial.getFechaDeteccion())
                    .processingTime(processingTime)
                    .probabilities(prediction.getProbabilities())
                    .build();

        } catch (Exception e) {
            log.error("Error procesando texto: {}", e.getMessage(), e);
            throw new RuntimeException("Error al procesar el texto: " + e.getMessage());
        }
    }

    private PythonPredictionResponse callPythonTextService(String text) {
        String url = pythonServiceUrl + "/predict-text";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        Map<String, String> requestBody = new HashMap<>();
        requestBody.put("text", text);

        HttpEntity<Map<String, String>> entity = new HttpEntity<>(requestBody, headers);

        try {
            ResponseEntity<PythonPredictionResponse> response = restTemplate.exchange(
                    url,
                    HttpMethod.POST,
                    entity,
                    PythonPredictionResponse.class);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            } else {
                throw new RuntimeException("Respuesta inválida del servicio Python (Texto)");
            }
        } catch (Exception e) {
            log.error("Error llamando al servicio Python (Texto): {}", e.getMessage());
            throw new RuntimeException("No se pudo conectar con el servicio de predicción de texto: " + e.getMessage());
        }
    }

    // Overloaded for backward compatibility
    private Historial saveToDatabase(PythonPredictionResponse prediction, long processingTime) {
        return saveToDatabase(prediction, processingTime, "IMAGE");
    }

    private Historial saveToDatabase(PythonPredictionResponse prediction, long processingTime, String tipo) {
        try {
            Historial historial = new Historial();
            historial.setEmocion(prediction.getEmotion());
            historial.setConfianza(prediction.getConfidence());
            historial.setTiempoProcesamiento(processingTime);
            historial.setProbabilidades(objectMapper.writeValueAsString(prediction.getProbabilities()));
            historial.setTipo(tipo);

            return historialRepository.save(historial);
        } catch (JsonProcessingException e) {
            log.error("Error serializando probabilidades: {}", e.getMessage());
            throw new RuntimeException("Error guardando en base de datos");
        }
    }
}