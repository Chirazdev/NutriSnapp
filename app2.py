import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import torch
from werkzeug.middleware.shared_data import SharedDataMiddleware

app = Flask(__name__)

# Path to the custom weights file
weights_path = r'C:\Users\Chiraz\testAPI\best.pt'

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)

# Dossier pour stocker les images téléchargées
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# Route pour gérer l'upload de l'image
@app.route('/', methods=['POST'])
def upload_file():
    # Vérifier si la requête contient un fichier
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    # Vérifier si le fichier est vide
    if file.filename == '':
        return redirect(request.url)

    # Vérifier si le fichier est un fichier image
    if file and allowed_file(file.filename):
        # Créer le dossier 'uploads' s'il n'existe pas
        uploads_folder = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(uploads_folder, exist_ok=True)

        # Sauvegarder le fichier dans le dossier d'uploads
        filepath = os.path.join(os.path.dirname(__file__), 'uploads', file.filename)
        file.save(filepath)

        # Charger l'image et effectuer une prédiction
        img = cv2.imread(str(filepath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img)

        # Dessiner les boîtes englobantes sur l'image
        annotated_img = results.render()[0]

        # Sauvegarder l'image annotée dans le dossier d'uploads
        annotated_filepath = os.path.join(os.path.dirname(__file__), 'uploads', 'annotated_' + file.filename)
        cv2.imwrite(annotated_filepath, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

        # Retourner le chemin de l'image annotée à la page
        return render_template('index.html', image_path=url_for('uploaded_file', filename='annotated_' + file.filename))

    return redirect(request.url)

# Fonction pour vérifier si l'extension du fichier est autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# Route pour servir les images téléchargées
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
