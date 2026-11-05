# app/processors/face_recognizer.py
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import config as config
from app.utils.cpu_threads import heavy_model_threads

# --- FIX #6: Removed hard-coded threshold ---
# SIMILARITY_THRESHOLD = 0.55 (This is now loaded from config in __init__)

class FaceRecognizer:
    """
    Handles student face identification using InsightFace.
    
    --- MODIFICATION (v2) ---
    This class is now initialized with a dictionary of *all* authorized students
    and their encodings. It finds the best match from this entire "known" pool.
    """
    
    def _limit_cpu_threads(self):
        """On CPU in async mode, reload the ONNX sessions with a capped thread pool.

        InsightFace doesn't pass session options through, and ONNX Runtime's
        default pool (one busy-waiting thread per core) starves the per-frame
        MediaPipe thread. See app/utils/cpu_threads.py.
        """
        threads = heavy_model_threads()
        if not threads:
            return
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        for model in self.model.models.values():
            if model.session.get_providers() == ['CPUExecutionProvider']:
                model.session = ort.InferenceSession(model.model_file, sess_options=opts,
                                                     providers=['CPUExecutionProvider'])
        self.logger.log_activity(f"Face Recognizer limited to {threads} CPU threads.")

    def __init__(self, logger, authorized_encodings: dict):
        """
        Initializes the recognizer for a multi-student session.
        
        Args:
            logger: The ActivityLogger instance.
            authorized_encodings (dict): A dictionary where keys are student names
                                         and values are lists of their encodings.
                                         e.g., {"Alice": [enc1, enc2], "Bob": [enc3]}
        """
        print(f"Initializing Face Recognizer (InsightFace/ONNX) for multi-student session.")
        self.logger = logger

        # --- FIX #6: Load threshold from config ---
        self.SIMILARITY_THRESHOLD = getattr(config, "FACE_SIMILARITY_THRESHOLD", 0.55)
        
        # --- FIX #5: Load model parameters from config ---
        det_size_cfg = getattr(config, "INSIGHTFACE_DET_SIZE", (640, 640))
        model_name = getattr(config, "INSIGHTFACE_MODEL_NAME", "buffalo_l")

        # --- FIX #5: GPU Fallback Logic ---
        gpu_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        cpu_providers = ['CPUExecutionProvider']
        # Only face detection and the ArcFace embedding are used; buffalo_l also
        # ships landmark and gender/age models that would run on every face.
        modules = getattr(config, "INSIGHTFACE_MODULES", ['detection', 'recognition'])

        try:
            # --- Attempt 1: Load with GPU support ---
            self.model = FaceAnalysis(
                name=model_name,
                providers=gpu_providers,
                allowed_modules=modules
            )
            self.model.prepare(ctx_id=0, det_size=det_size_cfg)
            # ONNX Runtime silently falls back to CPU when CUDA isn't usable, so
            # report the provider that is actually running the model.
            in_use = self.model.models['recognition'].session.get_providers()
            device = "GPU (CUDA)" if 'CUDAExecutionProvider' in in_use else "CPU"
            print(f"InsightFace model ({model_name}, {modules}) running on {device}.")
            self.logger.log_activity(f"Face Recognizer running on {device} (providers: {in_use}).")

        except Exception as e_gpu:
            print(f"WARNING: Failed to load InsightFace with GPU ({e_gpu}).")
            print("--- Falling back to CPU-only mode. ---")
            self.logger.log_alert("GPU_INIT_FAILURE", f"Failed to load InsightFace with GPU ({e_gpu}). Falling back to CPU.")
            
            try:
                # --- Attempt 2: Load with CPU-only support ---
                self.model = FaceAnalysis(
                    name=model_name,
                    providers=cpu_providers,
                    allowed_modules=modules
                )
                self.model.prepare(ctx_id=-1, det_size=det_size_cfg)
                print(f"InsightFace model ({model_name}) loaded with CPU-only support.")
                self.logger.log_activity("Face Recognizer loaded with CPU-only support.")
                
            except Exception as e_cpu:
                print(f"FATAL: Failed to load InsightFace model on CPU: {e_cpu}")
                self.logger.log_alert("FATAL_ERROR", f"FATAL: Failed to load InsightFace on CPU: {e_cpu}")
                raise # Re-raise the exception to stop the application
        # --- END FIX ---
        self._limit_cpu_threads()

        # --- Multi-Student Encoding Loader (Unchanged) ---
        self.known_encodings_normalized = [] # A flat list of all encodings
        self.known_names = []                # A parallel list of names for each encoding
        self.authorized_names_set = set(authorized_encodings.keys()) # For fast lookups

        if not authorized_encodings:
            msg = "Failed to initialize recognizer: No authorized encodings provided."
            self.logger.log_alert("REGISTRATION_FAILURE", msg)
            raise ValueError(msg)
            
        for name, encodings_list in authorized_encodings.items():
            if not encodings_list:
                self.logger.log_alert("REGISTRATION_WARNING", f"Student {name} has no encodings in database.")
                continue
                
            for enc in encodings_list:
                enc_array = np.array(enc)
                norm = np.linalg.norm(enc_array)
                if norm == 0:
                    continue
                
                self.known_encodings_normalized.append(enc_array / norm)
                self.known_names.append(name)
        
        if not self.known_names:
            msg = "Failed to initialize recognizer: No valid encodings could be loaded."
            self.logger.log_alert("REGISTRATION_FAILURE", msg)
            raise ValueError(msg)

        self.logger.log_activity(f"Recognizer active for {len(self.authorized_names_set)} students with {len(self.known_names)} total encoding(s).")
        self.logger.log_core_debug(f"Face similarity threshold set to: {self.SIMILARITY_THRESHOLD}")

    def get_authorized_names_set(self):
        """Helper for the FaceProcessor to know who is authorized."""
        return self.authorized_names_set

    def identify_faces(self, frame_bgr):
        """
        Detects and identifies all faces in a frame.
        Compares each detected face against the *entire* pool of authorized encodings.
        """
        if not self.known_names:
            return [], []

        identified_names = []
        detected_boxes = []

        try:
            faces = self.model.get(frame_bgr)

            for face in faces:
                new_embedding_raw = face.embedding 
                
                norm = np.linalg.norm(new_embedding_raw)
                if norm == 0:
                    continue
                new_embedding = new_embedding_raw / norm
                
                best_similarity = -1.0
                best_match_index = -1

                for i, known_embedding in enumerate(self.known_encodings_normalized):
                    similarity = np.dot(new_embedding, known_embedding)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_index = i
                
                final_name = "Unknown"
                # --- FIX #6: Use the class variable ---
                if best_similarity >= self.SIMILARITY_THRESHOLD:
                    final_name = self.known_names[best_match_index]
                
                identified_names.append(final_name)
                detected_boxes.append(face.bbox.astype(int))
                
                self.logger.log_similarity_debug(
                    f"Best similarity: {best_similarity:.4f}", 
                    person_name=final_name
                )

        except Exception as e:
            self.logger.log_alert("FACE_RECOGNIZER_ERROR", f"Error in identify_faces: {e}")
            return [], []

        return detected_boxes, identified_names