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
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Obtenez le chemin absolu du dossier actuel
current_dir = os.path.dirname(os.path.abspath(__file__))

# Définissez le chemin relatif vers le dossier d'images
images_dir = os.path.join(current_dir, 'images')


def load_model():
    print("Load model...")
    start_time = time.time()
    model = create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=1)

    # Charger l'état du modèle à partir du fichier checkpoint
    checkpoint_path = os.path.join(current_dir, '../model.pth')  # Utilisez le chemin absolu
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    model.eval()
    end_time = time.time()
    print(f"The model got loaded in {end_time - start_time} seconds")
    return model


def predict(image_id: str) -> float:

    image_path = f"{image_id}.jpg"
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    img = Image.open(os.path.join(images_dir, image_path)).convert('RGB')
    img_processed = preprocess(img)
    img_batch = img_processed.unsqueeze(0)
    prediction = model(img_batch)
    return prediction.item()


@app.route('/prediction-pet-score', methods=['POST'])
def home():
    if request.is_json:
        data = request.get_json()
        print(data)
        img_path = data.get('img_path')
        logging.debug("Received request with image path: %s", img_path)  # Ajoutez cette ligne pour afficher le chemin de l'image
        if img_path:
            score = predict(img_path)
            result = {"score": score}
            return jsonify(result), 200
        else:
            return jsonify({"error": "Image path not provided"}), 400
    else:
        return jsonify({"error": "Request must be JSON"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
