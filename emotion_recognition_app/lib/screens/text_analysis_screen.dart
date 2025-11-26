import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../models/emotion_response.dart';

class TextAnalysisScreen extends StatefulWidget {
  const TextAnalysisScreen({super.key});

  @override
  State<TextAnalysisScreen> createState() => _TextAnalysisScreenState();
}

class _TextAnalysisScreenState extends State<TextAnalysisScreen> {
  final TextEditingController _textController = TextEditingController();
  bool _isLoading = false;
  EmotionResponse? _result;
  String? _error;

  Future<void> _analyzeText() async {
    if (_textController.text.trim().isEmpty) return;

    setState(() {
      _isLoading = true;
      _error = null;
      _result = null;
    });

    try {
      final result = await ApiService.recognizeTextEmotion(_textController.text);
      setState(() {
        _result = result;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  String _getEmojiForEmotion(String emotion) {
    switch (emotion.toLowerCase()) {
      case 'joy':
      case 'happy':
      case 'felicidad':
        return '😄';
      case 'sadness':
      case 'sad':
      case 'tristeza':
        return '😢';
      case 'anger':
      case 'angry':
      case 'enojo':
        return '😠';
      case 'fear':
      case 'miedo':
        return '😨';
      case 'surprise':
      case 'sorpresa':
        return '😲';
      case 'disgust':
      case 'asco':
        return '🤢';
      default:
        return '😐';
    }
  }

  Color _getColorForEmotion(String emotion) {
    switch (emotion.toLowerCase()) {
      case 'joy':
        return Colors.yellow.shade700;
      case 'sadness':
        return Colors.blue;
      case 'anger':
        return Colors.red;
      case 'fear':
        return Colors.purple;
      case 'surprise':
        return Colors.orange;
      case 'disgust':
        return Colors.green;
      default:
        return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Análisis de Texto'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _textController,
              decoration: const InputDecoration(
                labelText: 'Escribe algo...',
                border: OutlineInputBorder(),
                hintText: 'Ej: Estoy muy feliz hoy',
              ),
              maxLines: 3,
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _isLoading ? null : _analyzeText,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
              child: _isLoading
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Analizar Emoción'),
            ),
            const SizedBox(height: 24),
            if (_error != null)
              Container(
                padding: const EdgeInsets.all(16),
                color: Colors.red.shade100,
                child: Text(
                  _error!,
                  style: TextStyle(color: Colors.red.shade900),
                ),
              ),
            if (_result != null) ...[
              Card(
                elevation: 4,
                child: Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: Column(
                    children: [
                      Text(
                        _getEmojiForEmotion(_result!.emotion),
                        style: const TextStyle(fontSize: 64),
                      ),
                      const SizedBox(height: 16),
                      Text(
                        _result!.emotion.toUpperCase(),
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: _getColorForEmotion(_result!.emotion),
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Confianza: ${(_result!.confidence * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(fontSize: 16, color: Colors.grey),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                'Probabilidades:',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              ..._result!.probabilities.entries.map((e) {
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    children: [
                      SizedBox(
                        width: 80,
                        child: Text(e.key),
                      ),
                      Expanded(
                        child: LinearProgressIndicator(
                          value: e.value,
                          backgroundColor: Colors.grey.shade200,
                          color: _getColorForEmotion(e.key),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text('${(e.value * 100).toStringAsFixed(1)}%'),
                    ],
                  ),
                );
              }).toList(),
            ],
          ],
        ),
      ),
    );
  }
}
