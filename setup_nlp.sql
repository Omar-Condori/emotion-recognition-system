-- Agregar columna 'tipo' a la tabla historial
-- Por defecto será 'IMAGE' para los registros existentes
ALTER TABLE historial ADD COLUMN tipo VARCHAR(20) DEFAULT 'IMAGE';

-- Opcional: Crear índice para búsquedas más rápidas por tipo
CREATE INDEX idx_historial_tipo ON historial(tipo);
