import 'package:flutter/material.dart';
import 'dart:io';
import '../models/emotion_response.dart';
import '../widgets/emotion_card.dart';

class ResultScreen extends StatelessWidget {
  final File imageFile;
  final EmotionResponse emotionResponse;

  const ResultScreen({
    Key? key,
    required this.imageFile,
    required this.emotionResponse,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final sortedProbabilities = emotionResponse.probabilities.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    return Scaffold(
      appBar: AppBar(
        title: const Text('Resultado'),
        centerTitle: true,
        elevation: 0,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Theme.of(context).primaryColor.withOpacity(0.1),
              Colors.white,
            ],
          ),
        ),
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Imagen capturada
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: Image.file(
                    imageFile,
                    height: 300,
                    fit: BoxFit.cover,
                  ),
                ),
              ),
              const SizedBox(height: 24),

              // Emoción principal
              EmotionCard(
                emotion: emotionResponse.emotion,
                probability: emotionResponse.confidence,
                isMain: true,
              ),
              const SizedBox(height: 24),

              // Título de otras emociones
              const Text(
                'Otras emociones detectadas:',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 16),

              // Lista de otras emociones
              ...sortedProbabilities.skip(1).map((entry) {
                return Padding(
                  padding: const EdgeInsets.only(bottom: 12.0),
                  child: EmotionCard(
                    emotion: entry.key,
                    probability: entry.value,
                  ),
                );
              }).toList(),

              const SizedBox(height: 24),

              // Información adicional
              if (emotionResponse.processingTime != null)
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text(
                              'Tiempo de procesamiento:',
                              style: TextStyle(fontWeight: FontWeight.w500),
                            ),
                            Text(
                              '${emotionResponse.processingTime} ms',
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                color: Colors.blue,
                              ),
                            ),
                          ],
                        ),
                        if (emotionResponse.id != null) ...[
                          const Divider(height: 24),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              const Text(
                                'ID de registro:',
                                style: TextStyle(fontWeight: FontWeight.w500),
                              ),
                              Text(
                                '#${emotionResponse.id}',
                                style: const TextStyle(
                                  fontWeight: FontWeight.bold,
                                  color: Colors.blue,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ],
                    ),
                  ),
                ),

              const SizedBox(height: 24),

              // Botón para volver
              ElevatedButton.icon(
                onPressed: () => Navigator.pop(context),
                icon: const Icon(Icons.arrow_back),
                label: const Text(
                  'Analizar otra imagen',
                  style: TextStyle(fontSize: 16),
                ),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}