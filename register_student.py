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

# Define the path for our database file
ENCODING_DB_PATH = "student_encodings.json"

def load_encodings(db_path):
    """Loads the encoding database from a JSON file."""
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {db_path} is corrupt. Starting fresh.")
                return {}
    return {}

def save_encodings(db_path, data):
    """Saves the encoding database to a JSON file."""
    with open(db_path, 'w') as f:
        json.dump(data, f, indent=2)

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

        # 3. Return as a list for JSON storage
        # The embedding is already a 1D numpy array, no squeeze needed
        return embedding.tolist(), len(faces)
        
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
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        arcface_model.prepare(ctx_id=0, det_size=(640, 640)) # Use a standard detection size
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load models: {e}")
        print("Please ensure 'insightface' is installed.")
        sys.exit(1)
    
    # 1. Load existing database
    db_data = load_encodings(ENCODING_DB_PATH)
    
    # 2. Get Student ID
    student_id = input("Enter Student ID (e.g., s1001): ").strip()
    if not student_id:
        print("Error: Student ID cannot be empty.")
        sys.exit(1)

    # 3. Check if student exists
    if student_id in db_data:
        student_name = db_data[student_id].get("name", "N/A")
        encoding_count = len(db_data[student_id].get("encodings", []))
        print(f"\nStudent ID '{student_id}' ({student_name}) already exists with {encoding_count} encoding(s).")
        
        print("  [1] Add a new face encoding to this student")
        print("  [2] Overwrite all existing data for this student (DANGEROUS)")
        print("  [3] Cancel")
        choice = input("Select an option (1-3): ").strip()
        
        if choice == '1':
            # --- Add new encoding ---
            image_path = input("Enter path to *new* student photo: ").strip()
            new_encoding, _ = process_image(image_path, arcface_model) # Updated call
            
            if new_encoding:
                if 'encodings' not in db_data[student_id] or not isinstance(db_data[student_id]['encodings'], list):
                    db_data[student_id]['encodings'] = []
                
                db_data[student_id]["encodings"].append(new_encoding)
                db_data[student_id]["status"] = "approved" # Automatically approve
                save_encodings(ENCODING_DB_PATH, db_data)
                print(f"\nSuccess! Added new encoding. '{student_name}' now has {len(db_data[student_id]['encodings'])} encodings.")
            else:
                print("Failed to add new encoding.")

        elif choice == '2':
            # --- Overwrite existing student ---
            print(f"Overwriting {student_name}...")
            student_name = input(f"Enter Student Name (for {student_id}): ").strip()
            image_path = input("Enter path to new student photo: ").strip()
            new_encoding, _ = process_image(image_path, arcface_model) # Updated call
            
            if new_encoding:
                db_data[student_id]["name"] = student_name # Also update name
                db_data[student_id]["encodings"] = [new_encoding] # Start a new list
                db_data[student_id]["status"] = "approved" # Automatically approve
                save_encodings(ENCODING_DB_PATH, db_data)
                print(f"\nSuccess! Overwrote student. '{student_name}' now has 1 encoding.")
            else:
                print("Failed to overwrite student.")

        else:
            print("Registration cancelled.")
            sys.exit(0)

    else:
        # --- Create new student ---
        print(f"\nCreating new registration for Student ID: {student_id}")
        student_name = input(f"Enter Student Name (for {student_id}): ").strip()
        if not student_name:
            print("Error: Student Name cannot be empty.")
            sys.exit(1)

        image_path = input("Enter path to student photo: ").strip()
        new_encoding, _ = process_image(image_path, arcface_model) # Updated call
        
        if new_encoding:
            db_data[student_id] = {
                "name": student_name,
                "encodings": [new_encoding], # <-- Store as a list
                "status": "approved" # Auto-approve students added via CLI
            }
            save_encodings(ENCODING_DB_PATH, db_data)
            print("\n--- Success! ---")
            print(f"Student '{student_name}' (ID: {student_id}) has been registered with 1 encoding.")
            print(f"Total students in database: {len(db_data)}")
        else:
            print("Registration failed.")
            sys.exit(1)

if __name__ == "__main__":
    main()