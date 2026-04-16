from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from faster_whisper import WhisperModel
import uuid

app = Flask(__name__)
CORS(app)

# Load the Whisper model once when the app starts (small model works well on Railway)
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})
    
    # Save uploaded file temporarily
    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        # Real transcription using faster-whisper
        segments, info = whisper_model.transcribe(
            filepath, 
            beam_size=5, 
            language="en", 
            vad_filter=True
        )
        
        # Combine all segments into clean lyrics
        lyrics = " ".join([segment.text.strip() for segment in segments])
        
        # Clean up the temporary file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            "success": True,
            "lyrics": lyrics,
            "detected_language": info.language
        })
        
    except Exception as e:
        # Clean up file if error occurs
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
