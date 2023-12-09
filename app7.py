import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import torch
from werkzeug.middleware.shared_data import SharedDataMiddleware
import base64
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

app = Flask(__name__)

# Path to the custom weights file
weights_path = r'C:\Users\Chiraz\testAPI\best.pt'

# Load the dataset
nutrition_df=pd.read_csv(r"C:\Users\Chiraz\Downloads\nutrients2.csv")

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

        # Convertir l'image annotée en format base64 pour afficher directement dans le HTML
        # Convertir l'image annotée en format base64 pour afficher directement dans le HTML
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode('utf-8')


        # Récupérer les informations de chaque boîte englobante
        boxes_info = [{'label': results.names[int(label)], 'confidence': confidence, 'coordinates': coord} for label, confidence, coord in zip(results.xyxy[0][:, 5].tolist(), results.xyxy[0][:, 4].tolist(), results.xyxy[0][:, :4].tolist())]
        
        # Get all detected class labels
        detected_classes = [results.names[int(label)] for label in results.xyxy[0][:, 5].tolist()]

        # Store nutritional values for each detected class
        all_nutritional_values = []
        
        # Check if any classes are detected
        if detected_classes:
           # Iterate through detected classes and find the first match in the nutrition DataFrame
           for detected_class in detected_classes:
               # Utiliser la fonction trouver_repas_proche pour trouver le repas le plus proche
               repas_proche = trouver_repas_proche(detected_class)
               print(repas_proche)

               # Filtrer le DataFrame nutrition_df pour obtenir les informations nutritionnelles du repas trouvé
               filtered_df = nutrition_df[nutrition_df['Food'] == repas_proche]

               nutritional_values = {}
               if not filtered_df.empty:
                  nutritional_values = filtered_df.iloc[0].to_dict()  # Use iloc[0] for the first row
                  all_nutritional_values.append({
                      'detected_class': detected_class,
                      'nutritional_values': nutritional_values,})
               else:
                   print("No nutritional information found for any detected class")
                   nutritional_values = {}
        else:
            # No classes detected, handle accordingly
            print("No classes detected in the image")
            detected_class = None
            nutritional_values = {}
        
        # Return annotated image, bounding box info, and nutritional values to the page
        return render_template('index.html', image=img_str, boxes_info=boxes_info, all_nutritional_values=all_nutritional_values, total_nutritional_values=calculate_total_nutritional_values(all_nutritional_values))

    return redirect(request.url)

def trouver_repas_proche(classe, seuil_similarity=0.7):
    # Utiliser le TfidfVectorizer pour convertir les descriptions de repas en vecteurs TF-IDF
    vectorizer = TfidfVectorizer()
    repas_vecteurs = vectorizer.fit_transform(nutrition_df['Food'].values.astype('U'))

    # Convertir la classe en vecteur TF-IDF
    classe_vecteur = vectorizer.transform([classe])

    # Calculer la similarité cosine entre la classe et chaque repas
    similarites = cosine_similarity(classe_vecteur, repas_vecteurs).flatten()

    # Trouver l'index du repas le plus proche
    index_repas_proche = np.argmax(similarites)

    # Récupérer le repas le plus proche et sa similarité
    repas_proche = nutrition_df.loc[index_repas_proche, 'Food']
    similarity = similarites[index_repas_proche]

    # Vérifier si la similarité est supérieure au seuil spécifié
    if similarity >= seuil_similarity:
        return repas_proche
    else:
        return "Aucun repas proche trouvé avec une similarité suffisante"


def calculate_total_nutritional_values(all_nutritional_values):
    # Initialize total nutritional values
    total_protein = 0
    total_calories = 0
    total_fiber = 0
    total_fat = 0
    total_carbs = 0

    # Calculate the total nutritional values
    for item in all_nutritional_values:
        nutritional_values = item.get('nutritional_values', {})
        total_protein += nutritional_values.get('Protein', 0)
        total_calories += nutritional_values.get('Calories', 0)
        total_fiber += nutritional_values.get('Fiber', 0)
        total_fat += nutritional_values.get('Fat', 0)
        total_carbs += nutritional_values.get('Carbs', 0)

    # Return the total nutritional values as a dictionary
    total_nutritional_values = {
        'Total Protein': total_protein,
        'Total Calories': total_calories,
        'Total Fiber': total_fiber,
        'Total Fat': total_fat,
        'Total Carbs': total_carbs,
    }

    return total_nutritional_values


# Fonction pour vérifier si l'extension du fichier est autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    app.run(debug=True)
