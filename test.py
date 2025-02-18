import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Liste des classes de fleurs
flower = [
    "astilbe",
    "bellflower",
    "black_eyed_susan",
    "calendula",
    "california_poppy",
    "carnation",
    "common_daisy",
    "coreopsis",
    "daffodil",
    "dandelion",
    "iris",
    "magnolia",
    "rose",
    "sunflower",
    "tulip",
    "water_lily"
]

# Charger le modèle pré-entraîné
model = tf.keras.models.load_model('flower_mobilenetv2_model2.h5')

# Fonction pour prédire l'image
def predict_flower(image_path, confidence_threshold=0.5):
    # Ouvrir l'image
    image = Image.open(image_path)
    
    # Redimensionner l'image à la taille attendue par le modèle (224x224)
    image = image.resize((224, 224))
    
    # Convertir l'image en tableau numpy et normaliser les valeurs
    image_array = np.array(image) / 255.0
    
    # Ajouter une dimension supplémentaire pour correspondre à l'entrée du modèle
    image_array = np.expand_dims(image_array, axis=0)
    
    # Faire la prédiction
    predictions = model.predict(image_array)
    
    # Obtenir la classe prédite et son score de confiance
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)  # Score de confiance de la prédiction
    
    # Vérifier si le score de confiance est supérieur au seuil
    if confidence < confidence_threshold:
        return "Image inconnue", confidence
    else:
        # Récupérer le nom de la fleur correspondante
        predicted_flower = flower[predicted_class[0]]
        return predicted_flower, confidence

# Chemin de l'image à tester
image_path = r"C:\Users\Adam\Downloads\assurence.jpeg"

# Faire la prédiction
result, confidence = predict_flower(image_path)

# Afficher le résultat
if result == "Image inconnue":
    print(f"Résultat : {result} (Confiance : {confidence:.2f})")
else:
    print(f"La fleur prédite est : {result} (Confiance : {confidence:.2f})")