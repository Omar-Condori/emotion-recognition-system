package com.emotion.recognition.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.time.LocalDateTime;

@Entity
@Table(name = "historial")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Historial {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String emocion;

    @Column(nullable = false)
    private Double confianza;

    @Column(name = "fecha_deteccion", nullable = false)
    private LocalDateTime fechaDeteccion;

    @Column(name = "tiempo_procesamiento")
    private Long tiempoProcesamiento;

    @Column(columnDefinition = "TEXT")
    private String probabilidades;

    @Column(length = 20)
    private String tipo; // "IMAGE" or "TEXT"

    @PrePersist
    protected void onCreate() {
        fechaDeteccion = LocalDateTime.now();
        if (tipo == null) {
            tipo = "IMAGE"; // Default for backward compatibility
        }
    }
}