from fastapi import FastAPI, File, UploadFile
from firebase_admin import credentials, firestore, initialize_app
from deepface import DeepFace
import numpy as np
import cv2
from utils import read_image_as_np

app = FastAPI()

# Firebase Init
cred = credentials.Certificate("serviceAccountKey.json")
initialize_app(cred)
db = firestore.client()

@app.post("/verify/")
async def verify_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_np = read_image_as_np(image_bytes)

    # Fetch all registered faces
    users_ref = db.collection("users").stream()
    for user in users_ref:
        user_dict = user.to_dict()
        ref_img = read_image_as_np(user_dict["img_bytes"])

        try:
            result = DeepFace.verify(img1_path=img_np, img2_path=ref_img, enforce_detection=False)
            if result["verified"]:
                return {"user_id": user.id, "verified": True}
        except Exception as e:
            continue

    return {"verified": False}
