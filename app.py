from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from PIL import Image
import numpy as np
import tensorflow as tf
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'} 

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

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

model = tf.keras.models.load_model('flower_mobilenetv2_model2.h5')

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


mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Ad.1234@",
    database="Fleure"
)
mycursor = mydb.cursor()
query = "SELECT * from plant" 
mycursor.execute(query)
results = mycursor.fetchall()
columns = [i[0] for i in mycursor.description]
df = pd.DataFrame(results, columns=columns)


mycursor.close()
mydb.close()

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")

def clean_and_extract_tags(text):
    doc = nlp(text.lower())
    tags = [token.text for token in doc if token.text.isalnum() and token.text not in STOP_WORDS]
    return ', '.join(tags)

columns_to_extract_tags_from = ['descriptions', 'descriptions_details', 'famille']

for column in columns_to_extract_tags_from:
    df[column] = df[column].apply(clean_and_extract_tags)

df['Tags'] = df[columns_to_extract_tags_from].apply(lambda row: ', '.join(row), axis=1)

def flower_recommendations(df, flowers_id, top_n=16):
    # Convertir les stop words en liste
    stop_words_list = list(STOP_WORDS)
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_list)
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(df['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    flowers_index = df[df['id'] == flowers_id].index[0]
    similar_flowers = list(enumerate(cosine_similarities_content[flowers_index]))
    similar_flowers = sorted(similar_flowers, key=lambda x: x[1], reverse=True)
    top_similar_flowers = similar_flowers[1:top_n + 1]
    recommended_flowers_indices = [x[0] for x in top_similar_flowers]
    recommended_flowers_details = df.iloc[recommended_flowers_indices][['id', 'nom', 'famille', 'descriptions', 'descriptions_details', 'vue', 'vente', 'image','prix']]

    return recommended_flowers_details

def top_flowers():
    average_ratings = df.groupby(
        ['id', 'nom', 'famille', 'descriptions', 'descriptions_details', 'vue', 'vente', 'image','prix'],
        as_index=False 
    )['vente'].mean().rename(columns={'vente': 'average_vente'})  

    top_rated_items = average_ratings.sort_values(by=['vue', 'average_vente'], ascending=False)

    rating_base_recommendation = top_rated_items.head(16)
    return rating_base_recommendation

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_flower_details(flower_name):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Ad.1234@",
        database="Fleure"
    )
    mycursor = mydb.cursor(dictionary=True)
    query = "SELECT * FROM plant WHERE nom LIKE %s"
    mycursor.execute(query, (f"%{flower_name}%",))
    result = mycursor.fetchall()
    mycursor.close()
    mydb.close()
    return result
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Faire la prédiction
        result, confidence = predict_flower(filepath)
        print(result, confidence)
        
        # Obtenir les détails de la fleur prédite
        flower_details = get_flower_details(result)
        
        # Supprimer le fichier après traitement (optionnel)
        os.remove(filepath)
        
        return jsonify({'result': result, 'confidence': float(confidence), 'details': flower_details}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400
@app.route('/')
def index():
    top_flower = top_flowers()
    return render_template('index2.html', top_flowers=top_flower.to_dict(orient='records'))



@app.route('/details/<int:plant_id>')
def plant_details(plant_id):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Ad.1234@",
        database="Fleure"
    )
    mycursor = mydb.cursor(dictionary=True)
    query = "SELECT * FROM plant WHERE id = %s"
    mycursor.execute(query, (plant_id,))
    plant_details = mycursor.fetchone()
    mycursor.close()
    mydb.close()
    recommendations = flower_recommendations(df, plant_id)
    return render_template('product_details.html', plant_details=plant_details, recommendations=recommendations.to_dict(orient='records'))

# Démarrer l'application
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')