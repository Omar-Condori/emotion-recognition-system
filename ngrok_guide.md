# Guía Completa: Exponer Backend Local con ngrok para Flutter (macOS)

Esta guía te explicará paso a paso cómo hacer que tu backend local (Python/Java) sea accesible desde cualquier dispositivo móvil usando **ngrok**.

## 1. Instalación de ngrok en macOS

Tienes dos opciones principales para instalar ngrok en tu Mac.

### Opción A: Usando Homebrew (Recomendada)
Si tienes Homebrew instalado, es la forma más rápida:

```bash
brew install ngrok/ngrok/ngrok
```

### Opción B: Descarga Directa
1. Ve a [ngrok.com/download](https://ngrok.com/download).
2. Descarga la versión para **macOS**.
3. Descomprime el archivo descargado.
4. Mueve el ejecutable a una carpeta en tu PATH (ej. `/usr/local/bin`) o ejecútalo directamente desde donde lo descomprimiste.

---

## 2. Crear Cuenta y Obtener AuthToken

Para usar ngrok sin restricciones de tiempo de sesión, necesitas una cuenta gratuita.

1. Regístrate en [dashboard.ngrok.com/signup](https://dashboard.ngrok.com/signup).
2. Una vez dentro, ve al menú lateral izquierdo y haz clic en **"Your Authtoken"**.
3. Verás una cadena larga de caracteres que empieza por `2...`. Ese es tu token.

---

## 3. Vincular tu Cuenta

Abre tu terminal y ejecuta el siguiente comando (reemplaza `<TU_TOKEN>` con el que copiaste):

```bash
ngrok config add-authtoken <TU_TOKEN>
```

*Esto guardará tu token en un archivo de configuración en tu Mac, por lo que no tendrás que ponerlo cada vez.*

---

## 4. Exponer tu Servidor Local

Supongamos que tu backend Python corre en el puerto **8000**. Para exponerlo a internet:

1. Asegúrate de que tu backend esté corriendo (`python main.py`).
2. En una **nueva pestaña** de terminal, ejecuta:

```bash
ngrok http 8000
```

### Opciones Útiles
- **Región**: Si estás en Sudamérica, puedes intentar usar la región `sa` para menor latencia (aunque `us` suele ser la predeterminada y estable).
  ```bash
  ngrok http 8000 --region sa
  ```
- **Inspección de Tráfico**: ngrok te da una interfaz web para ver las peticiones que llegan. Abre `http://127.0.0.1:4040` en tu navegador mientras ngrok corre.

---

## 5. Entendiendo la URL Pública

Al ejecutar el comando, verás algo así en tu terminal:

```text
Forwarding                    https://a1b2-c3d4.ngrok-free.app -> http://localhost:8000
```

- **URL Pública**: `https://a1b2-c3d4.ngrok-free.app`
- **¿Qué significa?**: Cualquier petición que se haga a esa URL desde internet (tu celular, el de un amigo, etc.) viajará a los servidores de ngrok, bajará por el túnel seguro hasta tu Mac y llegará a tu `localhost:8000`.

---

## 6. Configurar Flutter para usar ngrok

Ahora debes decirle a tu app Flutter que use esta nueva URL en lugar de `localhost` o `10.0.2.2`.

### Paso 1: Archivo de Constantes (Básico)
La forma más rápida es tener un archivo de configuración.

Crea o edita `lib/config.dart`:

```dart
class Config {
  // COPIA Y PEGA AQUÍ LA URL QUE TE DIO NGROK (sin la barra final /)
  static const String baseUrl = "https://a1b2-c3d4.ngrok-free.app";
}
```

### Paso 2: Usar la URL en tu Servicio
En tu `api_service.dart`:

```dart
import 'config.dart';
import 'package:http/http.dart' as http;

class ApiService {
  static Future<void> getData() async {
    // Usamos la URL de ngrok
    final url = Uri.parse('${Config.baseUrl}/predict-text');
    
    final response = await http.get(url);
    // ... resto del código
  }
}
```

### Paso 3: Opción Avanzada (.env)
Para proyectos más serios, usa el paquete `flutter_dotenv`.

1. Agrega `flutter_dotenv` a `pubspec.yaml`.
2. Crea un archivo `.env` en la raíz:
   ```env
   API_URL=https://a1b2-c3d4.ngrok-free.app
   ```
3. Cárgalo en `main.dart` y úsalo:
   ```dart
   await dotenv.load(fileName: ".env");
   String url = dotenv.env['API_URL'] ?? 'http://localhost:8000';
   ```

---

## 7. Recomendaciones y Mantenimiento

- **Terminal Separada**: Mantén la terminal de ngrok **siempre abierta**. Si la cierras, el túnel se rompe y la URL deja de funcionar.
- **Cambio de URL**: En el plan gratuito, **cada vez que reinicias ngrok, la URL cambia**. Tendrás que copiar la nueva URL y actualizar tu código Flutter cada vez.
  - *Tip*: Si no quieres recompilar la app a cada rato, puedes poner un campo de texto oculto en tu app para pegar la URL manualmente durante el desarrollo.

---

## 8. Advertencias y Límites (Plan Gratuito)

- **URL Dinámica**: Como se mencionó, la URL cambia al reiniciar.
- **Límite de Conexiones**: Tiene un límite de conexiones por minuto (suficiente para desarrollo, pero no para producción masiva).
- **Seguridad**: Cualquiera con la URL puede acceder a tu backend. No compartas la URL públicamente si tienes datos sensibles expuestos sin autenticación.
- **Página de Advertencia**: A veces, al abrir la URL de ngrok por primera vez en un navegador, muestra una advertencia de "ngrok-skip-browser-warning". Si tu API recibe peticiones JSON puras, esto no suele afectar, pero tenlo en cuenta si algo falla.

---

## 9. Alternativas Permanentes

Si te cansas de cambiar la URL cada vez, considera:

1. **ngrok Plan Pago**: Te permite reservar dominios fijos (ej. `mi-api.ngrok.io`).
2. **Deploy en la Nube (Gratis/Barato)**:
   - **Render / Railway / Fly.io**: Puedes subir tu contenedor Docker o código Python/Java y te dan una URL `https` permanente (ej. `mi-app.onrender.com`). Esto es lo ideal para demos estables.
   - **VPS (DigitalOcean/AWS)**: Más control, pero requiere configurar Linux, Nginx, SSL, etc.

---

### Resumen Rápido para tu Día a Día

1. Abre terminal 1: `python main.py` (Tu backend)
2. Abre terminal 2: `ngrok http 8000`
3. Copia la URL `https://....ngrok-free.app`
4. Pégala en tu código Flutter (`api_service.dart` o `config.dart`).
5. Ejecuta tu app en el celular.
