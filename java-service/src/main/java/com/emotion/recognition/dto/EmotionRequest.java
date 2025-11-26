package com.emotion.recognition.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class EmotionRequest {
    @NotBlank(message = "La imagen en Base64 es requerida")
    private String image;
}