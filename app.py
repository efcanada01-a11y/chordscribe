from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from faster_whisper import WhisperModel
import os

app = Flask(__name__, static_folder='.')
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Whisper model (downloads on first run - may take time)
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Real transcription using faster-whisper
    segments, info = whisper_model.transcribe(filepath, beam_size=5, language="en")
    lyrics = " ".join([seg.text for seg in segments])

    return jsonify({
        "success": True,
        "lyrics": lyrics,
        "detected_language": info.language
    })

if __name__ == '__main__':
    print("🚀 Chordscribe with Real Transcription running at http://localhost:5000")
    app.run(port=5000, debug=True)