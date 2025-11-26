class EmotionResponse {
  final int? id;
  final String emotion;
  final double confidence;
  final String? timestamp;
  final int? processingTime;
  final Map<String, double> probabilities;

  EmotionResponse({
    this.id,
    required this.emotion,
    required this.confidence,
    this.timestamp,
    this.processingTime,
    required this.probabilities,
  });

  factory EmotionResponse.fromJson(Map<String, dynamic> json) {
    return EmotionResponse(
      id: json['id'],
      emotion: json['emotion'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      timestamp: json['timestamp'],
      processingTime: json['processingTime'],
      probabilities: (json['probabilities'] as Map<String, dynamic>)
          .map((key, value) => MapEntry(key, (value as num).toDouble())),
    );
  }

  String get emotionEmoji {
    switch (emotion.toLowerCase()) {
      case 'happy':
        return '😊';
      case 'sad':
        return '😢';
      case 'angry':
        return '😠';
      case 'fear':
        return '😨';
      case 'surprise':
        return '😮';
      case 'disgust':
        return '🤢';
      case 'neutral':
        return '😐';
      default:
        return '🤔';
    }
  }

  String get confidencePercentage => '${(confidence * 100).toStringAsFixed(1)}%';
}