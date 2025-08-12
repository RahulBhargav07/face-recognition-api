from fastapi import FastAPI, File, UploadFile, Form
from recognize import recognize_face, add_face
import uvicorn

app = FastAPI()

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    result = await recognize_face(file)
    return result

@app.post("/add-face/")
async def add_face_endpoint(
    name: str = Form(...), 
    file: UploadFile = File(...)
):
    result = await add_face(name, file)
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
