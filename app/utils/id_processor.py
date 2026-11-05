# app/utils/id_processor.py
import cv2
import numpy as np
import pytesseract
import re
from PIL import Image

# --- MODIFICATION: Removed all torch, MTCNN, and insightface imports ---
# These models are now "injected" (passed in) rather than loaded here.

class IDProcessor:
    """
    Handles all processing for ID card verification:
    1.  Image preprocessing for OCR.
    2.  Text extraction using Tesseract.
    3.  Face detection and embedding on the ID card using InsightFace.
    """
    
    # --- MODIFICATION: __init__ now accepts the 'arcface_model' ---
    def __init__(self, logger, arcface_model):
        self.logger = logger
        
        # --- MODIFICATION: Store the model passed from get_processors ---
        self.arcface_model = arcface_model
        
        # --- REMOVED ---
        # All model loading logic (device, MTCNN, FaceAnalysis) is gone.
        
        print("ID Processor initialized and received shared FaceAnalysis model.")

    def _preprocess_for_ocr(self, image_bgr):
        """
        Applies image filters to improve OCR accuracy, 
        especially for small or blurry text.
        (This function is unchanged)
        """
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            denoised = cv2.medianBlur(dilated, 3)
            return denoised
        
        except Exception as e:
            self.logger.log_alert("OCR_PREPROCESS_ERROR", f"Error in preprocessing: {e}")
            return None

    def extract_text(self, image_bgr):
        """
        Runs OCR on the preprocessed image to extract all text.
        (This function is unchanged)
        """
        preprocessed_image = self._preprocess_for_ocr(image_bgr)
        if preprocessed_image is None:
            return ""

        try:
            config_str = '--psm 6'
            text = pytesseract.image_to_string(preprocessed_image, config=config_str)
            return text
        except Exception as e:
            self.logger.log_alert("TESSERACT_ERROR", f"Error running Tesseract: {e}")
            print(f"FATAL: Tesseract error. Is it installed and in PATH? Error: {e}")
            return ""

    # --- MODIFICATION: This function is completely rewritten for accuracy ---
    def find_id_card_photo_embedding(self, image_bgr):
        """
        Finds the face on the ID card and returns its embedding
        using the injected InsightFace model.

        This new pipeline (detect, align, embed) is more robust
        for low-quality or small faces on ID cards.

        Args:
            image_bgr (np.ndarray): The captured image of the ID card.

        Returns:
            np.ndarray or None: The face embedding (as a numpy array) of the face on the ID, 
                                or None if no suitable face is found.
        """
        try:
            # --- NEW EFFICIENT LOGIC ---
            # The 'get' method handles detection, alignment, and embedding all at once.
            faces = self.arcface_model.get(image_bgr)

            if not faces:
                self.logger.log_activity("ID_VERIFY: No face photo found on ID card image.")
                return None

            # 'faces' is sorted by detection score, so faces[0] is the best one.
            best_face = faces[0]
            embedding = best_face.embedding
            
            # The embedding is already a 1D numpy array, ready to be used.
            return embedding

        except Exception as e:
            self.logger.log_alert("ID_FACE_FIND_ERROR", f"Error finding/encoding face on ID: {e}")
            return None

    def verify_info(self, extracted_text, student_id, student_name):
        """
        Checks if the student's ID and Name are reasonably present in the extracted OCR text.
        (This function is unchanged)
        """
        text_lower = extracted_text.lower()
        id_normalized = re.sub(r'[\s-]', '', student_id).lower()
        text_no_spaces = re.sub(r'[\s-]', '', text_lower)
        id_found = id_normalized in text_no_spaces

        name_parts = student_name.lower().split()
        if not name_parts:
            return False

        name_found_all_parts = all(part in text_lower for part in name_parts)
        name_found_first_part = name_parts[0] in text_lower
        name_found = name_found_all_parts or name_found_first_part

        if not id_found: self.logger.log_activity(f"ID_VERIFY_FAIL: ID '{student_id}' not found in OCR text.")
        if not name_found: self.logger.log_activity(f"ID_VERIFY_FAIL: Name '{student_name}' not found in OCR text.")

        return id_found and name_found