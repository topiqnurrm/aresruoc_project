from flask import Flask, request, jsonify
from src.emotion_detector import emotion_predictor

app = Flask(__name__)

@app.route('/emotion', methods=['POST'])
def detect_emotion():
    data = request.get_json()
    text = data.get('text', '')
    result = emotion_predictor(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/emotion', methods=['POST'])
def detect_emotion():
    try:
        data = request.get_json()
        text = data.get('text', '')
        result = emotion_predictor(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
