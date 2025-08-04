import os
from PIL import Image
from imgbeddings import imgbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ibed = imgbeddings()

RECOGNITION_THRESHOLD = 0.7

def load_known_faces(folder_path="known_people"):
    known_embeddings = []
    known_names = []
    for file in os.listdir(folder_path):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            name = os.path.splitext(file)[0].replace("_", " ").title()
            img = Image.open(os.path.join(folder_path, file))
            embedding = ibed.to_embeddings(img)[0]
            known_embeddings.append(embedding)
            known_names.append(name)
    return known_embeddings, known_names

def recognize_person(uploaded_img, known_embeddings, known_names):
    img = Image.open(uploaded_img)
    embedding = ibed.to_embeddings(img)[0].reshape(1, -1)
    similarities = cosine_similarity(embedding, known_embeddings)
    index = np.argmax(similarities)
    confidence = similarities[0][index]
    if confidence > RECOGNITION_THRESHOLD:
        return known_names[index], float(confidence)
    return "Unknown", float(confidence)
