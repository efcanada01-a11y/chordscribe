from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
from faster_whisper import WhisperModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
import uuid
import gc

app = Flask(__name__)
CORS(app)

whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

UPLOAD_FOLDER = "uploads"
PDF_FOLDER = "pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

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
        segments, info = whisper_model.transcribe(
            filepath, 
            beam_size=5, 
            language="en", 
            vad_filter=False
        )
        
        lyrics_list = [segment.text.strip() for segment in segments if segment.text.strip()]
        lyrics = "\n".join(lyrics_list) if lyrics_list else "No clear vocals detected."
        
        # Generate PDF
        pdf_filename = f"{uuid.uuid4()}.pdf"
        pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
        generate_pdf(pdf_path, file.filename, lyrics)
        
        # Clean up audio
        if os.path.exists(filepath):
            os.remove(filepath)
        gc.collect()
        
        return jsonify({
            "success": True,
            "lyrics": lyrics,
            "pdf_filename": pdf_filename
        })
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        gc.collect()
        return jsonify({"success": False, "error": str(e)})

def generate_pdf(pdf_path, song_name, lyrics):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 80, "Chordscribe")
    c.setFont("Helvetica", 18)
    c.drawString(50, height - 120, f"Song: {song_name}")
    
    # Lyrics with chords simulation
    c.setFont("Helvetica-Bold", 14)
    y = height - 180
    common_chords = ["C", "Am", "F", "G", "Em", "Dm"]
    chord_idx = 0
    
    lines = lyrics.split('\n')
    for line in lines:
        if not line.strip():
            y -= 20
            continue
            
        chord = common_chords[chord_idx % len(common_chords)]
        c.drawString(70, y, f"[{chord}]")
        c.setFont("Helvetica", 12)
        wrapped = simpleSplit(line.strip(), "Helvetica", 12, 450)
        for wline in wrapped:
            c.drawString(140, y, wline)
            y -= 18
        chord_idx += 1
        y -= 10
        
        if y < 100:
            c.showPage()
            y = height - 80
    
    c.save()

@app.route('/download/pdf/<filename>')
def download_pdf(filename):
    return send_file(os.path.join(PDF_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
