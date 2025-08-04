import mediapipe as mp
import cv2
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

ibed = imgbeddings()
mp_face_detection = mp.solutions.face_detection

def detect_face(image_np):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        results = detector.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    if results and results.detections:
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w, _ = image_np.shape
        x, y = int(bbox.xmin * w), int(bbox.ymin * h)
        w2, h2 = int(bbox.width * w), int(bbox.height * h)
        return image_np[y:y+h2, x:x+w2]
    return None

def get_embedding(pil_img):
    return ibed.to_embeddings(pil_img)[0]

def match_face(embedding, known_embeddings, known_names, threshold=0.7):
    embedding = embedding.reshape(1, -1)
    sims = cosine_similarity(embedding, known_embeddings)
    idx = int(np.argmax(sims))
    score = float(sims[0][idx])
    if score >= threshold:
        return known_names[idx], score
    return "Unknown", 0.0
