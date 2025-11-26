package com.emotion.recognition.controller;

import com.emotion.recognition.dto.EmotionRequest;
import com.emotion.recognition.dto.EmotionResponse;
import com.emotion.recognition.entity.Historial;
import com.emotion.recognition.service.EmotionService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/emotions")
@RequiredArgsConstructor
@Slf4j
@CrossOrigin(origins = "*")
public class EmotionController {

    private final EmotionService emotionService;

    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        Map<String, String> response = new HashMap<>();
        response.put("status", "UP");
        response.put("service", "emotion-recognition-java");
        return ResponseEntity.ok(response);
    }

    @PostMapping("/recognize")
    public ResponseEntity<?> recognizeEmotion(@Valid @RequestBody EmotionRequest request) {
        try {
            log.info("Recibida solicitud de reconocimiento de emoción (Imagen)");
            EmotionResponse response = emotionService.processEmotion(request);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            log.error("Error en reconocimiento: {}", e.getMessage());
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("error", e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    @PostMapping("/recognize-text")
    public ResponseEntity<?> recognizeTextEmotion(
            @Valid @RequestBody com.emotion.recognition.dto.TextEmotionRequest request) {
        try {
            log.info("Recibida solicitud de reconocimiento de emoción (Texto)");
            EmotionResponse response = emotionService.processTextEmotion(request);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            log.error("Error en reconocimiento de texto: {}", e.getMessage());
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("error", e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    @GetMapping("/history")
    public ResponseEntity<List<Historial>> getHistory() {
        List<Historial> historial = emotionService.getHistorial();
        return ResponseEntity.ok(historial);
    }

    @GetMapping("/history/emotion/{emotion}")
    public ResponseEntity<List<Historial>> getHistoryByEmotion(@PathVariable String emotion) {
        List<Historial> historial = emotionService.getHistorialByEmocion(emotion);
        return ResponseEntity.ok(historial);
    }
}