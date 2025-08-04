from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io
import os

from face_utils import detect_face, get_embedding, match_face
from firebase_utils import init_firebase

app = FastAPI()
db = init_firebase()

known_names, known_embeddings = [], []

def load_known_faces():
    from imgbeddings import imgbeddings
    ib = imgbeddings()
    folder = "known_people"
    if not os.path.isdir(folder):
        return
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(folder, fname))
            emb = ib.to_embeddings(img)[0]
            name = os.path.splitext(fname)[0].replace("_", " ").title()
            known_names.append(name)
            known_embeddings.append(emb)

@app.on_event("startup")
async def startup_event():
    load_known_faces()

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data))
    arr = np.array(img.convert("RGB"))
    face = detect_face(arr)
    if face is None:
        return {"status": "no_face"}
    pil_face = Image.fromarray(face)
    emb = get_embedding(pil_face)
    name, confidence = match_face(emb, known_embeddings, known_names)
    return {"name": name, "confidence": confidence}
