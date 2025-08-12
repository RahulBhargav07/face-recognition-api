import firebase_admin
from firebase_admin import credentials, firestore
from imgbeddings import imgbeddings
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
import io

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Load embeddings from Firebase (at startup)
print("Loading known embeddings from Firebase...")
docs = db.collection("persons_mediapipe").stream()
known_names, known_embeddings = [], []
for doc in docs:
    data = doc.to_dict()
    known_names.append(data["name"])
    known_embeddings.append(np.array(data["face_embedding"]))
print(f"Loaded {len(known_names)} embeddings.")

ibed = imgbeddings()
THRESHOLD = 0.7

async def recognize_face(file):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        query_embedding = ibed.to_embeddings(image)[0].reshape(1, -1)

        similarities = cosine_similarity(query_embedding, np.array(known_embeddings))
        best_match_index = np.argmax(similarities)
        confidence = similarities[0][best_match_index]

        if confidence > THRESHOLD:
            name = known_names[best_match_index]
        else:
            name = "Unknown"

        return {
            "name": name,
            "confidence": round(float(confidence), 4)
        }

    except Exception as e:
        return {"error": str(e)}

async def add_face(name, file):
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Create embedding
        embedding = ibed.to_embeddings(image)[0].tolist()

        # Store in Firestore
        db.collection("persons_mediapipe").document(name).set({
    "name": name,
    "face_embedding": embedding
})

        # Update local memory (optional)
        known_names.append(name)
        known_embeddings.append(np.array(embedding))

        return {"status": "success", "name": name}
    
    except Exception as e:
        return {"error": str(e)}
