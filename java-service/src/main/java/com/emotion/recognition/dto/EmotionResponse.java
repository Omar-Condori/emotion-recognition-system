package com.emotion.recognition.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.time.LocalDateTime;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class EmotionResponse {
    private Long id;
    private String emotion;
    private Double confidence;
    private LocalDateTime timestamp;
    private Long processingTime;
    private Map<String, Double> probabilities;
}