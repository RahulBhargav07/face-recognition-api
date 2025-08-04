import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os

ibed = imgbeddings()
face_db = {}  # In-memory cache {name: (embedding, details)}

def enroll_face(image: Image.Image, name: str, details: str):
    emb = ibed.to_embeddings(image)[0]
    face_db[name] = (emb, details)
    return {"name": name, "details": details, "status": "enrolled"}

def recognize_face(image: Image.Image, threshold: float = 0.7):
    if not face_db:
        return {"status": "no known faces"}

    emb = ibed.to_embeddings(image)[0].reshape(1, -1)
    best_match = {"name": "Unknown", "confidence": 0.0}

    for name, (known_emb, details) in face_db.items():
        sim = cosine_similarity(emb, known_emb.reshape(1, -1))[0][0]
        if sim > best_match["confidence"]:
            best_match = {"name": name, "confidence": float(sim), "details": details}

    if best_match["confidence"] < threshold:
        best_match["name"] = "Unknown"

    return best_match
