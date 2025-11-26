package com.emotion.recognition.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TextEmotionRequest {
    @NotBlank(message = "El texto es requerido")
    private String text;
}
