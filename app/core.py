# app/core.py
import cv2
import config as config
from datetime import datetime
import threading
import queue
import time
import numpy as np

# --- Processor Imports ---
from app.processors.object_detector import ObjectDetector
from app.processors.mediapipe_processor import MediaPipeProcessor
from app.processors.face_processor import FaceProcessor
from app.processors.face_recognizer import FaceRecognizer
# --- Refactored Imports ---
from app.processors.person_tracker import PersonTracker
from app.processors.proctoring_analyzer import ProctoringAnalyzer

# --- Drawing Utility ---
from app.utils.drawing import draw_detections, draw_status_panel, draw_face_annotations, draw_contraband

"""
This file is the main orchestrator for the proctoring application.
... (docstring) ...
"""

class ProctoringApp:
    # --- MODIFIED: __init__ signature updated (Phase 4) ---
    # Added exam_context=None
    def __init__(self, logger, face_recognizer, authorized_student_names: list, exam_context=None,
                 rois_data=None):
        self.logger = logger
        
        # --- Initialize Processors ---
        self.face_recognizer = face_recognizer
        self.mediapipe_processor = MediaPipeProcessor()
        self.face_processor = FaceProcessor(face_recognizer=self.face_recognizer)
        self.object_detector = ObjectDetector(logger=self.logger)
        
        # --- Initialize NEW Refactored Processors ---
        self.person_tracker = PersonTracker(logger=logger)
        
        # --- MODIFIED: Pass exam_context to the analyzer (Phase 4) ---
        self.proctoring_analyzer = ProctoringAnalyzer(
            logger=logger,
            authorized_student_names=authorized_student_names,
            exam_context=exam_context,
            rois_data=rois_data
        )
        
        # --- Set Loggers for Processors ---
        self.face_processor.set_alert_logger(self.logger)
        self.mediapipe_processor.set_alert_logger(self.logger)
        
        # --- Threading & Video Capture ---
        self.frame_queue = queue.Queue(maxsize=2)
        self.processed_frame_lock = threading.Lock()
        self.latest_processed_frame = None
        self.stop_event = threading.Event()

        print("Starting processing thread...")
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.start()
        print(f"Attempting to open camera at index: {config.CAMERA_INDEX}")
        self.video_capture = None
        max_retries = 5
        for i in range(max_retries):
            self.video_capture = cv2.VideoCapture(config.CAMERA_INDEX)
            if self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if ret and frame is not None:
                    self.logger.log_activity(f"Webcam opened (attempt {i+1}).")
                    break
                else:
                    self.logger.log_activity(f"Camera opened but failed to read frame (attempt {i+1}/{max_retries}).")
                    self.video_capture.release()
                    time.sleep(1)
            else:
                self.logger.log_activity(f"Camera init failed (attempt {i+1}/{max_retries}).")
                time.sleep(1)
        
        if not self.video_capture or not self.video_capture.isOpened():
            self.logger.log_alert("CAMERA_ERROR", f"FATAL: Cannot open webcam at index {config.CAMERA_INDEX}.")
            raise IOError(f"FATAL: Cannot open webcam at index {config.CAMERA_INDEX}.")

    def process_frame(self, frame):
        h, w, _ = frame.shape
        
        # --- 1. Run Detections (Objects and Faces) ---
        # (ObjectDetector will be updated to run on an interval)
        persons, contraband_items = self.object_detector.detect(frame) 
        # (FaceProcessor now runs on an interval and returns smoothed alerts)
        face_results = self.face_processor.detect_faces(frame) 
        
        # --- 2. Run MediaPipe Landmark Detection ---
        list_of_mp_results = []
        if face_results['boxes']:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                list_of_mp_results = self.mediapipe_processor.process(frame_rgb)
            except Exception as e:
                self.logger.log_core_debug(f"[FATAL] FATAL ERROR during conversion or mediapipe processing: {e}")
                self.logger.log_alert("MEDIAPIPE_ERROR", f"Failed to convert frame to RGB or process: {e}")

        # --- 3. Update Person Tracker ---
        # This links faces to landmarks and assigns persistent IDs
        tracked_persons = self.person_tracker.update_tracks(face_results, list_of_mp_results, w, h, frame)

        # --- 4. Run Proctoring Analysis ---
        # Get the smoothed alert flag from the face_processor's results
        unauthorized_alert = face_results.get('unauthorized_alert_active', False)
        
        # This checks for liveness, pose alerts, and global alerts
        final_classification = self.proctoring_analyzer.analyze_frame(
            frame, 
            tracked_persons, 
            contraband_items,
            unauthorized_alert
        )
            
        # --- 5. Draw Everything ---
        # Note: 'persons' from ObjectDetector is drawn, but 'tracked_persons'
        # is used for face boxes. This is correct, as 'persons' is just
        # a raw detection, while 'tracked_persons' is the fused result.
        draw_detections(frame, persons) 
        draw_contraband(frame, contraband_items)
        
        # This new function draws the detailed boxes/landmarks for *tracked* faces
        draw_face_annotations(
            frame, 
            tracked_persons
        )
        
        draw_status_panel(frame, final_classification)
        return frame

    # --- Processing Worker Thread (Unchanged) ---
    def _processing_worker(self):
        print("Processing worker started.")
        while not self.stop_event.is_set():
            try:
                frame_to_process = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            processed_frame = self.process_frame(frame_to_process)

            with self.processed_frame_lock:
                self.latest_processed_frame = processed_frame
        
        print("Processing worker stopping.")

    # --- Main Application Loop (Unchanged) ---
    def run(self):
        self.logger.log_activity("Surveillance system started.")
        while True:
            if self.stop_event.is_set():
                break 

            ret, frame = self.video_capture.read()
            if not ret:
                print("Error: Failed to grab frame. Exiting.")
                self.logger.log_alert("CAMERA_ERROR", "Failed to grab frame.")
                break

            live_frame = cv2.flip(frame, 1)

            try:
                self.frame_queue.put_nowait(live_frame)
            except queue.Full:
                pass

            processed_frame_to_show = None
            with self.processed_frame_lock:
                if self.latest_processed_frame is not None:
                    processed_frame_to_show = self.latest_processed_frame

            display_frame = processed_frame_to_show if processed_frame_to_show is not None else live_frame
            cv2.imshow("Production-Ready Proctoring System", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.logger.log_activity("User pressed 'q'. Shutting down.")
                break
        self.cleanup()

    # --- Cleanup (Unchanged, but includes previous bugfix) ---
    def cleanup(self):
        print("Shutting down system... signaling worker thread...")
        self.logger.log_core_debug("Cleanup sequence started. Signaling worker thread.") 
        self.stop_event.set()

        try:
            self.processing_thread.join(timeout=5.0)
        except Exception:
            try: self.processing_thread.join()
            except Exception: pass

        print("Worker thread stopped. Releasing resources...")
        self.logger.log_core_debug("Worker thread stopped. Releasing resources.") 

        try:
            if hasattr(self, "video_capture") and self.video_capture is not None:
                self.video_capture.release()
                self.logger.log_core_debug("Video capture released.") 
        except Exception as e:
            print(f"[CLEANUP] Error releasing video capture: {e}")
            self.logger.log_core_debug(f"Error releasing video capture: {e}") 

        try:
            if hasattr(self, "mediapipe_processor") and self.mediapipe_processor is not None:
                close_fn = getattr(self.mediapipe_processor, "close", None)
                if callable(close_fn):
                    close_fn()
                    self.logger.log_core_debug("MediaPipe processor closed.") 
                try: del self.mediapipe_processor
                except Exception: pass
        except Exception as e:
            print(f"[CLEANUP] Error closing MediaPipe processor: {e}")
            self.logger.log_core_debug(f"Error closing MediaPipe processor: {e}") 

        try:
            cv2.destroyAllWindows()
            self.logger.log_core_debug("OpenCV windows destroyed.") 
        except Exception:
            pass