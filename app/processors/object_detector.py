# app/processors/object_detector.py
import torch
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
import config as config
import cv2
from app.utils.cpu_threads import heavy_model_threads

class ObjectDetector:
    """
    Encapsulates the RT-DETR model from the Hugging Face transformers library.
    Handles model loading, preprocessing, inference, and post-processing.
    Detects persons and a list of predefined contraband items.
    
    --- MODIFICATION (Fix #3) ---
    Now includes an internal frame counter to run detection on an interval,
    reducing the performance impact from running the heavy model every frame.
    """
    def __init__(self,logger):
        self.logger = logger
        print("Initializing Object Detector (RT-DETR)...")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.log_core_debug(f"Object Detector will use device: {self.device}")
            threads = heavy_model_threads() if self.device == "cpu" else None
            if threads:
                torch.set_num_threads(threads)
                self.logger.log_core_debug(f"Object Detector limited to {threads} CPU threads.")

            self.processor = RTDetrImageProcessor.from_pretrained(config.HF_MODEL_NAME)
            self.model = RTDetrForObjectDetection.from_pretrained(config.HF_MODEL_NAME).to(self.device)

            self.classes_to_detect = config.CLASSES_TO_DETECT
            self.contraband_list = config.CONTRABAND_CLASSES
            
            # --- FIX #3: State for interval-based detection ---
            self.frame_count = 0
            # Use a sensible default (e.g., 10) if not in config
            self.detection_interval = getattr(config, "OBJECT_DETECTION_INTERVAL", 10) 
            self.last_persons = []
            self.last_contraband_items = []
            # --- END FIX ---

            self.logger.log_core_debug("RT-DETR model and processor loaded successfully.")
        except Exception as e:
            self.logger.log_core_debug(f"FATAL ERROR: Failed to load the RT-DETR model: {e}")
            raise

    def detect(self, frame):
        """
        Performs object detection on a single video frame.
        
        --- MODIFICATION (Fix #3) ---
        This function now only runs the full model inference every N frames
        (defined by self.detection_interval). On other frames, it returns
        the cached results from the last detection.
        """
        
        self.frame_count += 1
        
        # --- FIX #3: Interval check ---
        if self.frame_count % self.detection_interval != 0:
            # Not an interval frame, return the cached results
            return self.last_persons, self.last_contraband_items
        # --- END FIX ---

        # --- It is an interval frame, run full detection ---
        self.logger.log_detection_debug("Running RT-DETR model inference (interval frame)")
        
        persons = []
        contraband_items = []
        
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=config.DETECTION_CONFIDENCE_THRESHOLD,
                target_sizes=target_sizes
            )[0]
            
            # --- Raw logging (unchanged) ---
            try:
                raw_detections = []
                for score, label_id in zip(results["scores"], results["labels"]):
                    class_name = self.model.config.id2label[label_id.item()]
                    raw_detections.append(f"{class_name} ({score.item():.2f})")
                
                if raw_detections:
                    self.logger.log_detection_debug(f"Raw detections: [{', '.join(raw_detections)}]")
                else:
                    self.logger.log_detection_debug("Raw detections: [None found]")
            except Exception as e:
                self.logger.log_core_debug(f"Error during raw detection logging: {e}")
            # --- End raw logging ---

            for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
                class_name = self.model.config.id2label[label_id.item()]

                if class_name in self.classes_to_detect:
                    item_data = {
                        'box': [round(i) for i in box.tolist()],
                        'score': score.item(),
                        'label': class_name
                    }
                    if class_name == 'person':
                        persons.append(item_data)
                    elif class_name in self.contraband_list:
                        contraband_items.append(item_data)
        
        except Exception as e:
            self.logger.log_alert("OBJECT_DETECTOR_ERROR", f"Error during RT-DETR inference: {e}")
            # On error, return the last known good detection to avoid flicker
            return self.last_persons, self.last_contraband_items

        # --- FIX #3: Cache the new results ---
        self.last_persons = persons
        self.last_contraband_items = contraband_items
        # --- END FIX ---

        return self.last_persons, self.last_contraband_items