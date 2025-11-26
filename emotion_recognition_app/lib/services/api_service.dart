import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../models/emotion_response.dart';

class ApiService {
  // Cambia esta URL según tu configuración
  // Para iOS Simulator: http://localhost:8080
  // Para Android Emulator: http://10.0.2.2:8080
  // Para dispositivo físico: http://TU_IP_LOCAL:8080
  static const String baseUrl = 'http://172.20.10.2:8080';

  static Future<String> _getBaseUrl() async {
    if (Platform.isAndroid) {
      return 'http://172.20.10.2:8080';
    } else {
      return baseUrl;
    }
  }

  static Future<EmotionResponse> recognizeEmotion(File imageFile) async {
    try {
      final url = await _getBaseUrl();

      // Leer imagen y convertir a Base64
      final bytes = await imageFile.readAsBytes();
      final base64Image = base64Encode(bytes);

      // Hacer petición al backend Java
      final response = await http.post(
        Uri.parse('$url/api/emotions/recognize'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'image': base64Image,
        }),
      ).timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(response.body);
        return EmotionResponse.fromJson(jsonResponse);
      } else {
        throw Exception('Error del servidor: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error al conectar con el servidor: $e');
    }
  }

  static Future<EmotionResponse> recognizeTextEmotion(String text) async {
    try {
      final url = await _getBaseUrl();

      final response = await http.post(
        Uri.parse('$url/api/emotions/recognize-text'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'text': text,
        }),
      ).timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(response.body);
        return EmotionResponse.fromJson(jsonResponse);
      } else {
        throw Exception('Error del servidor: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error al conectar con el servidor: $e');
    }
  }

  static Future<List<dynamic>> getHistory() async {
    try {
      final url = await _getBaseUrl();

      final response = await http.get(
        Uri.parse('$url/api/emotions/history'),
        headers: {
          'Content-Type': 'application/json',
        },
      ).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        return jsonDecode(response.body) as List<dynamic>;
      } else {
        throw Exception('Error al obtener historial');
      }
    } catch (e) {
      throw Exception('Error al conectar con el servidor: $e');
    }
  }

  static Future<bool> checkHealth() async {
    try {
      final url = await _getBaseUrl();

      final response = await http.get(
        Uri.parse('$url/api/emotions/health'),
      ).timeout(const Duration(seconds: 5));

      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}