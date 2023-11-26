import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import torch

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
        # Sauvegarder le fichier dans le dossier d'uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Charger l'image et effectuer une prédiction
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img)

        # Afficher les résultats
        print(results)
        
        # Retourner les résultats à la page
        return render_template('index.html', results=results.xyxy[0])

    return redirect(request.url)

# Fonction pour vérifier si l'extension du fichier est autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    app.run(debug=True)
