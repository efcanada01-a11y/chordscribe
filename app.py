from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from faster_whisper import WhisperModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
import uuid
import gc

app = Flask(__name__)
CORS(app)
app.config['JWT_SECRET_KEY'] = 'super-secret-key-change-in-production'
jwt = JWTManager(app)

whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

UPLOAD_FOLDER = "uploads"
PDF_FOLDER = "pdfs"
MIDI_FOLDER = "midi"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(MIDI_FOLDER, exist_ok=True)

# Simple SQLite setup
def get_db():
    conn = sqlite3.connect('chordscribe.db')
    conn.row_factory = sqlite3.Row
    return conn

with get_db() as conn:
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        credits INTEGER DEFAULT 5
    )''')

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"success": False, "error": "Email and password required"}), 400
    
    hashed = generate_password_hash(password)
    
    try:
        with get_db() as conn:
            conn.execute("INSERT INTO users (email, password, credits) VALUES (?, ?, 5)", (email, hashed))
            conn.commit()
        return jsonify({"success": True, "message": "Account created"})
    except Exception as e:
        return jsonify({"success": False, "error": "Email already exists"}), 400

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    with get_db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    
    if user and check_password_hash(user['password'], password):
        access_token = create_access_token(identity=str(user['id']))
        return jsonify({"success": True, "token": access_token, "email": email})
    
    return jsonify({"success": False, "error": "Invalid credentials"}), 401

@app.route('/api/transcribe', methods=['POST'])
@jwt_required()
def transcribe():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        segments, info = whisper_model.transcribe(filepath, beam_size=5, language="en", vad_filter=False)
        lyrics = "\n".join([segment.text.strip() for segment in segments if segment.text.strip()])
        
        pdf_filename = f"{uuid.uuid4()}.pdf"
        pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
        generate_pdf(pdf_path, file.filename, lyrics)
        
        midi_filename = f"{uuid.uuid4()}.mid"
        midi_path = os.path.join(MIDI_FOLDER, midi_filename)
        generate_simple_midi(midi_path)
        
        if os.path.exists(filepath):
            os.remove(filepath)
        gc.collect()
        
        return jsonify({
            "success": True,
            "lyrics": lyrics or "No lyrics detected.",
            "pdf_filename": pdf_filename,
            "midi_filename": midi_filename
        })
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        gc.collect()
        return jsonify({"success": False, "error": str(e)}), 500

def generate_pdf(pdf_path, song_name, lyrics):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 80, "Chordscribe")
    c.setFont("Helvetica", 18)
    c.drawString(50, height - 120, f"Song: {song_name}")
    y = height - 200
    common_chords = ["C", "Am", "F", "G", "Em", "Dm"]
    chord_idx = 0
    for line in lyrics.split('\n'):
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
        y -= 15
        if y < 100:
            c.showPage()
            y = height - 100
    c.save()

def generate_simple_midi(midi_path):
    with open(midi_path, "wb") as f:
        f.write(b'MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\x60MTrk\x00\x00\x00\x0A\x00\xFF\x51\x03\x0F\x42\x40\x00\xFF\x2F\x00')

@app.route('/download/pdf/<filename>')
def download_pdf(filename):
    return send_file(os.path.join(PDF_FOLDER, filename), as_attachment=True)

@app.route('/download/midi/<filename>')
def download_midi(filename):
    return send_file(os.path.join(MIDI_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
