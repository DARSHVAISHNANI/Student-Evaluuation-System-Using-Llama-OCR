from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from Ollama_OCR_Pipeline.OCR_Processors import OCR_Processors
from Ollama_OCR_Pipeline.Answer_Evalute import AnswerEvaluator
from Ollama_OCR_Pipeline.ResponseEvaluator import ResponseEvaluator
import tensorflow as tf
import os
from flask_cors import CORS
from datetime import datetime
import nltk
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True)  # Enable CORS for frontend access

# Define upload and processed folders
UPLOAD_FOLDER = 'uploads/images/'
PROCESSED_FOLDER = 'uploads/processed/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Function to check allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to generate a unique filename
def unique_filename(filename):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name, ext = os.path.splitext(filename)
    return f"{name}_{timestamp}{ext}"

# OCR processing function
def process_image(image_path):
    with tf.device('/GPU:0'):
        ocr = OCR_Processors(model_name='llama3.2-vision:11b')
        batch_results = ocr.process_batch(image_path)
        for _, text in batch_results['results'].items():
            return text.replace('\n', '')
    return None

# Text evaluation function
def evaluate_text(reference_answer, student_answer, total_marks=10):
    weights = {
        "BLEU Score": 0.05,
        "SBERT Similarity": 0.95,
        "Redundancy Penalty": -0.2
    }
    
    evaluator = ResponseEvaluator(model_name='sentence-transformers/all-MiniLM-L6-v2')
    scores = evaluator.evaluate_response(reference_answer, student_answer, weights)
    marks = evaluator.calculate_marks(scores, total_marks, weights)
    feedback = evaluator.generate_feedback(scores, marks, total_marks)
    
    return marks, feedback

# API endpoint for file upload and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    if 'reference_answer' not in request.form:
        return jsonify({'error': 'No reference answer provided'}), 400

    uploaded_file = request.files['file']
    reference_answer = request.form['reference_answer']

    if uploaded_file and allowed_file(uploaded_file.filename):
        filename = secure_filename(unique_filename(uploaded_file.filename))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        uploaded_file.save(file_path)
        extracted_text = process_image(file_path)

        marks, feedback = evaluate_text(reference_answer, extracted_text)

        processed_filename = f"processed_{filename}.txt"
        processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        with open(processed_file_path, 'w') as f:
            f.write(extracted_text)

        return jsonify({
            'extracted_text': extracted_text,
            'marks': marks,
            'feedback': feedback,
            'processed_file': processed_filename
        })

    return jsonify({'error': 'Invalid file type'}), 400

# API endpoint to download processed file
@app.route('/processed/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)