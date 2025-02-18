from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

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


def flower_recommendations(df, flowers_id, top_n=10):
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


@app.route('/')
def index():
    top_flower = top_flowers()
    return render_template('index2.html', top_flowers=top_flower.to_dict(orient='records'))

# Démarrer l'application
if __name__ == '__main__':
    app.run(debug=True)