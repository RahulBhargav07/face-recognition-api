from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io
import cv2
import os

from face_utils import detect_face, get_embedding, match_face
from firebase_utils import init_firebase

app = FastAPI()
db = init_firebase()

known_names = []
known_embeddings = []

def load_known_faces():
    global known_names, known_embeddings
    from imgbeddings import imgbeddings
    from PIL import Image
    ibed = imgbeddings()
    folder = "known_people"
    if not os.path.exists(folder):
        return
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, fname)
            img = Image.open(path)
            embedding = ibed.to_embeddings(img)[0]
            name = os.path.splitext(fname)[0].replace("_", " ").title()
            known_names.append(name)
            known_embeddings.append(embedding)

@app.on_event("startup")
async def startup_event():
    load_known_faces()

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    contents = await file.read()
    image_np = np.array(Image.open(io.BytesIO(contents)))
    face = detect_face(image_np)
    if face is None:
        return {"status": "no_face"}

    pil_face = Image.fromarray(face)
    embedding = get_embedding(pil_face)
    name, confidence = match_face(embedding, known_embeddings, known_names)
    return {"name": name, "confidence": float(confidence)}
