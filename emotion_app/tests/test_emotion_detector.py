import unittest
from src.emotion_detector import emotion_predictor

class TestEmotionDetector(unittest.TestCase):
    def test_emotion_predictor_valid_input(self):
        result = emotion_predictor("I am happy today!")
        self.assertIn('status', result)
        self.assertEqual(result['status'], 'success')

    def test_emotion_predictor_empty_input(self):
        result, status_code = emotion_predictor("")
        self.assertEqual(status_code, 400)
        self.assertIn('error', result)

if __name__ == '__main__':
    unittest.main()
