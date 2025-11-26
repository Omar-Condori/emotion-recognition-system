import 'package:flutter/material.dart';

class EmotionCard extends StatelessWidget {
  final String emotion;
  final double probability;
  final bool isMain;

  const EmotionCard({
    Key? key,
    required this.emotion,
    required this.probability,
    this.isMain = false,
  }) : super(key: key);

  String get emoji {
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

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: isMain ? 8 : 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: isMain
            ? BorderSide(color: Theme.of(context).primaryColor, width: 2)
            : BorderSide.none,
      ),
      child: Padding(
        padding: EdgeInsets.all(isMain ? 24.0 : 16.0),
        child: Column(
          children: [
            Text(
              emoji,
              style: TextStyle(fontSize: isMain ? 64 : 40),
            ),
            const SizedBox(height: 8),
            Text(
              emotion,
              style: TextStyle(
                fontSize: isMain ? 24 : 18,
                fontWeight: isMain ? FontWeight.bold : FontWeight.w500,
              ),
            ),
            const SizedBox(height: 8),
            LinearProgressIndicator(
              value: probability,
              minHeight: 8,
              backgroundColor: Colors.grey[300],
              valueColor: AlwaysStoppedAnimation<Color>(
                isMain ? Theme.of(context).primaryColor : Colors.blue,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              '${(probability * 100).toStringAsFixed(1)}%',
              style: TextStyle(
                fontSize: isMain ? 20 : 16,
                fontWeight: FontWeight.bold,
                color: Theme.of(context).primaryColor,
              ),
            ),
          ],
        ),
      ),
    );
  }
}