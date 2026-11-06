# register_student.py
import json
import os
import sys
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis # Use InsightFace for both detection and embeddings
from PIL import Image

# --- REMOVED torch and facenet_pytorch imports ---

import app.utils.database as db
from app.utils import students_db


def process_image(image_path, arcface_model):
    """
    Loads an image, finds the face, and generates an ArcFace embedding
    using the modern InsightFace .get() method.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None, 0
        
    try:
        print("Loading image and finding face...")
        # Load image with OpenCV (InsightFace prefers BGR)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Error: Could not read image file at {image_path}")
            return None, 0

        # 1. Detect face and get embedding using InsightFace
        # .get() handles detection, alignment, and encoding
        faces = arcface_model.get(img_bgr)
        
        if not faces:
            print("Error: No face found in the provided image. Please use a clear photo.")
            return None, 0
            
        if len(faces) > 1:
            print(f"Warning: Multiple faces ({len(faces)}) found. Using the largest one (index 0).")
        
        # 2. Get the embedding from the best face
        embedding = faces[0].embedding
        if embedding is None:
            print("Error: Could not generate embedding for the face.")
            return None, 0

        return embedding, len(faces)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, 0

def main():
    """
    A command-line tool to register or update student encodings.
    """
    print("--- Student Encoding Registration (InsightFace/ONNX Model) ---")

    # --- Load Models ---
    print("Loading InsightFace model...")
    try:
        # --- REMOVED MTCNN ---
        
        # Load InsightFace Model
        arcface_model = FaceAnalysis(
            name='buffalo_l', 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition']  # only the embedding is used
        )
        arcface_model.prepare(ctx_id=0, det_size=(640, 640)) # Use a standard detection size
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load models: {e}")
        print("Please ensure 'insightface' is installed.")
        sys.exit(1)
    
    db.init_database()

    student_id = input("Enter Student ID (e.g., s1001): ").strip()
    if not student_id:
        print("Error: Student ID cannot be empty.")
        sys.exit(1)

    existing = students_db.get_student(student_id)
    if existing:
        print(f"\nStudent ID '{student_id}' ({existing['name']}) already exists "
              f"with {existing['encoding_count']} encoding(s), status '{existing['status']}'.")
        print("  [1] Add a new face encoding to this student")
        print("  [2] Overwrite all existing data for this student (DANGEROUS)")
        print("  [3] Cancel")
        choice = input("Select an option (1-3): ").strip()
        if choice == '1':
            encoding, _ = process_image(input("Enter path to *new* student photo: ").strip(), arcface_model)
            if encoding is None:
                sys.exit("Failed to add new encoding.")
            count = students_db.add_encodings(student_id, [encoding])
            students_db.set_status(student_id, "approved", reviewed_by="register_student.py")
            print(f"\nSuccess! '{existing['name']}' now has {count} encodings.")
            return
        if choice != '2':
            print("Registration cancelled.")
            return
        print(f"Overwriting {existing['name']}...")

    student_name = input(f"Enter Student Name (for {student_id}): ").strip()
    if not student_name:
        sys.exit("Error: Student Name cannot be empty.")
    encoding, _ = process_image(input("Enter path to student photo: ").strip(), arcface_model)
    if encoding is None:
        sys.exit("Registration failed.")
    # Students added by an administrator from the command line are approved directly
    students_db.save_registration(student_id, student_name, [encoding], extracted_id_text="CLI registration",
                                  status="approved")
    print("\n--- Success! ---")
    print(f"Student '{student_name}' (ID: {student_id}) has been registered with 1 encoding.")


if __name__ == "__main__":
    main()