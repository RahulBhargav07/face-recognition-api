from fastapi import FastAPI, File, UploadFile
from firebase_config import init_firebase
from recognition import load_known_faces, recognize_person
import shutil
import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"

app = FastAPI()
db = init_firebase()
known_embeddings, known_names = load_known_faces()

@app.get("/")
def home():
    return {"message": "Face Recognition API is running."}

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    name, confidence = recognize_person(path, known_embeddings, known_names)
    os.remove(path)
    return {"name": name, "confidence": confidence}
