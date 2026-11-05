# core/models.py
import streamlit as st
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import insightface
from insightface.app import FaceAnalysis

@st.cache_resource
def get_models():
    """Loads and caches the AI models."""
    print("Loading AI Models...")
    arcface_model = None
    face_mesh_model = None
    
    try:
        arcface_model = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition']  # only the embedding is used
        )
        arcface_model.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace ArcFace (buffalo_l) model loaded.")
    except Exception as e:
        print(f"Could not load InsightFace model: {e}")

    try:
        face_mesh_model = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("MediaPipe Face Mesh model loaded.")
    except Exception as e:
        print(f"Could not load MediaPipe model: {e}")

    print("Model loading complete.")
    return arcface_model, face_mesh_model