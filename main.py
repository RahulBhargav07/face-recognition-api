from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from face_utils import enroll_face, recognize_face
from PIL import Image
import io

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Face Recognition API is running"}

@app.post("/enroll")
async def enroll(
    name: str = Form(...),
    details: str = Form(...),
    image: UploadFile = File(...)
):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    result = enroll_face(img, name, details)
    return result

@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    result = recognize_face(img)
    return result
