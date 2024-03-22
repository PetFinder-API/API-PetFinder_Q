import time
import os
import torch

from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from timm import create_model
from torchvision import transforms
import logging

app = Flask(__name__)
CORS(app, origins='http://localhost:3000')
logging.basicConfig(level=logging.DEBUG)

# Obtenez le chemin absolu du dossier actuel
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Définissez le chemin absolu vers le dossier d'images
images_dir = os.path.join(parent_dir, 'images')

# Déclaration de la variable modèle en dehors de la fonction load_model
model = None

def load_model():
    print("Load model...")
    start_time = time.time()
    model = create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=1)

    # Charger l'état du modèle à partir du fichier checkpoint
    checkpoint_path = os.path.join(os.path.dirname(current_dir), 'model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    model.eval()
    end_time = time.time()
    print(f"The model got loaded in {end_time - start_time} seconds")
    return model

model = load_model()

def predict(image_id: str) -> float:
    image_path = os.path.join(images_dir, f"{image_id}")

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert('RGB')
    img_processed = preprocess(img)
    img_batch = img_processed.unsqueeze(0)
    prediction = model(img_batch)
    return prediction.item()

@app.route('/prediction-pet-score', methods=['POST'])
def home():
    if request.is_json:
        data = request.get_json()
        print("Received JSON data:", data)

        img_path = data.get('img_path')
        print("Image path received:", img_path)

        if img_path:
            # Construire le chemin absolu de l'image
            abs_img_path = os.path.join(images_dir, img_path)
            print("Absolute image path:", abs_img_path)

            # Vérifier si le fichier image existe
            if os.path.exists(abs_img_path):
                print("Image file exists.")
                # Prédire le score
                score = predict(img_path)
                result = {"score": round(score, 2)}

                # Renvoyer le score prédit dans la réponse JSON
                return jsonify(result), 200
            else:
                print("Image file does not exist.")
                return jsonify({"error": "Image not found"}), 404
        else:
            print("Image path not provided.")
            return jsonify({"error": "Image path not provided"}), 400
    else:
        print("Request is not JSON.")
        return jsonify({"error": "Request must be JSON"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
