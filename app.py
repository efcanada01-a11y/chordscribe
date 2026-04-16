from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from faster_whisper import WhisperModel
import uuid

app = Flask(__name__)
CORS(app)

# Use a slightly better setting for music/songs
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
    
    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        # Better settings for songs (disable aggressive VAD, allow more segments)
        segments, info = whisper_model.transcribe(
            filepath, 
            beam_size=5, 
            language="en", 
            vad_filter=False,           # Turn off strict VAD for music
            word_timestamps=False
        )
        
        lyrics_list = [segment.text.strip() for segment in segments if segment.text.strip()]
        lyrics = " ".join(lyrics_list)
        
        # Fallback message if nothing detected
        if not lyrics:
            lyrics = "No clear vocals detected in this audio. Try a song with louder singing or clearer vocals."
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            "success": True,
            "lyrics": lyrics,
            "detected_language": info.language
        })
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)