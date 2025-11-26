package com.emotion.recognition.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PythonPredictionResponse {
    private String emotion;
    private Double confidence;
    private Map<String, Double> probabilities;
}