from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from faster_whisper import WhisperModel

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"})
    
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        segments, info = whisper_model.transcribe(filepath, beam_size=5, language="en", vad_filter=False)
        lyrics = "\n".join([segment.text.strip() for segment in segments if segment.text.strip()])
        if not lyrics:
            lyrics = "No lyrics detected in this audio."
    except Exception as e:
        lyrics = f"Transcription failed: {str(e)}"
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
    
    return jsonify({"success": True, "lyrics": lyrics})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
