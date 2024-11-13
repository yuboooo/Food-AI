from flask import Flask, request
import os

app = Flask(__name__)

# Directory where images will be saved
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return "File uploaded successfully", 200

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=6712)
