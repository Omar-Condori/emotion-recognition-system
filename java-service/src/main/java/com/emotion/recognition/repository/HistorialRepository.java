package com.emotion.recognition.repository;

import com.emotion.recognition.entity.Historial;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface HistorialRepository extends JpaRepository<Historial, Long> {
    List<Historial> findByEmocion(String emocion);
    List<Historial> findByFechaDeteccionBetween(LocalDateTime inicio, LocalDateTime fin);
    List<Historial> findTop10ByOrderByFechaDeteccionDesc();
}