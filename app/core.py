# app/core.py
import cv2
import config as config
import threading
import queue
import time
from collections import deque

# --- Processor Imports ---
from app.processors.object_detector import ObjectDetector
from app.processors.mediapipe_processor import MediaPipeProcessor
from app.processors.face_processor import FaceProcessor
# --- Refactored Imports ---
from app.processors.person_tracker import PersonTracker
from app.processors.proctoring_analyzer import ProctoringAnalyzer

# --- Drawing Utility ---
from app.utils.drawing import draw_detections, draw_status_panel, draw_face_annotations, draw_contraband
from app.video_source import FileSource, open_source

"""
This file is the main orchestrator for the proctoring application.
... (docstring) ...
"""

class ProctoringApp:
    # --- MODIFIED: __init__ signature updated (Phase 4) ---
    # Added exam_context=None
    def __init__(self, logger, face_recognizer, authorized_student_names: list, exam_context=None,
                 rois_data=None, video_source=None, display=True, publish_jpeg=False):
        """
        video_source: a CameraSource/FileSource (see app/video_source.py), or None for
                      the webcam at config.CAMERA_INDEX.
        display:      show the annotated feed in an OpenCV window (the CLI); False when
                      the engine runs as a service behind the web API.
        publish_jpeg: keep the latest annotated frame as JPEG bytes for streaming.
        """
        self.logger = logger
        self.display = display
        self.publish_jpeg = publish_jpeg
        # Open the video source first, so a missing camera fails before any threads start
        self.video_source = video_source if video_source is not None else open_source(logger=logger)
        
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
        self._cleaned_up = False

        # --- Published output (for the web API) ---
        self.last_classification = "Normal"
        self._frame_cond = threading.Condition()
        self._jpeg = None
        self._jpeg_seq = 0
        self._status = {}
        self._processed_times = deque(maxlen=60)
        self.frames_processed = 0

        # --- Heavy-model inference thread ---
        # RT-DETR and InsightFace take hundreds of ms per frame on CPU. Run in
        # the processing thread they stall it, and MediaPipe misses blinks
        # during the stall. In async mode they run in their own thread on the
        # newest frame, and every processed frame uses their latest results.
        self.async_inference = getattr(config, "HEAVY_MODELS_ASYNC", True)
        self.inference_stop = threading.Event()
        self._inference_lock = threading.Lock()
        self._inference_frame = None
        self._inference_results = ([], [], {'boxes': [], 'names': [], 'face_present': False,
                                            'student_present': False, 'unauthorized_alert_active': False})
        self.inference_count = 0
        self.inference_thread = None
        if self.async_inference:
            self.object_detector.detection_interval = 1   # the thread itself sets the pace
            self.face_processor.detection_interval = 1
            self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
            self.inference_thread.start()

        print("Starting processing thread...")
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.start()

    def _inference_worker(self):
        """Runs RT-DETR + InsightFace on the newest frame, as often as they can keep up."""
        min_interval = getattr(config, "HEAVY_MODEL_MIN_INTERVAL_SEC", 0.2)
        while not self.inference_stop.is_set():
            with self._inference_lock:
                frame, self._inference_frame = self._inference_frame, None
            if frame is None:
                self.inference_stop.wait(0.005)
                continue
            started = time.time()
            try:
                persons, contraband_items = self.object_detector.detect(frame)
                face_results = self.face_processor.detect_faces(frame)
            except Exception as e:
                self.logger.log_alert("INFERENCE_ERROR", f"Heavy-model inference failed: {e}")
                continue
            with self._inference_lock:
                self._inference_results = (persons, contraband_items, face_results)
                self.inference_count += 1
            remaining = min_interval - (time.time() - started)
            if remaining > 0:
                self.inference_stop.wait(remaining)   # cap the rate to spare the GPU/CPU

    def process_frame(self, frame):
        h, w, _ = frame.shape

        # --- 1. Detections (Objects and Faces) ---
        if self.async_inference:
            with self._inference_lock:
                self._inference_frame = frame.copy()  # newest frame for the inference thread
                persons, contraband_items, face_results = self._inference_results
        else:
            # Synchronous mode: both run here every OBJECT/FACE_DETECTION_INTERVAL frames
            persons, contraband_items = self.object_detector.detect(frame)
            face_results = self.face_processor.detect_faces(frame)

        # Evidence requested during analysis is saved after drawing (see logger)
        getattr(self.logger, "begin_frame", lambda: None)()

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
        getattr(self.logger, "end_frame", lambda f: None)(frame)
        self.last_classification = final_classification
        return frame

    # --- Processing Worker Thread ---
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
            self._publish(processed_frame)

        print("Processing worker stopping.")

    # --- Published output ---------------------------------------------------
    def _publish(self, frame):
        """Updates the status snapshot and (if enabled) the JPEG for streaming."""
        now = time.perf_counter()
        self._processed_times.append(now)
        self.frames_processed += 1
        status = self._build_status()
        jpeg = None
        if self.publish_jpeg:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg = buf.tobytes() if ok else None
        with self._frame_cond:
            self._status = status
            if jpeg is not None:
                self._jpeg = jpeg
                self._jpeg_seq += 1
            self._frame_cond.notify_all()

    def _build_status(self):
        times = self._processed_times
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
        people = []
        for p in list(self.person_tracker.tracked_persons.values()):
            mp = p.mp_result or {}
            people.append({
                "track_id": p.id,
                "name": p.get_display_name(),
                "is_photo": bool(p.is_spoofing),
                "liveness": p.liveness_state,
                "alerts": list(getattr(p, "current_alerts", [])),
                "yaw": round(float(mp["yaw"]), 1) if mp.get("yaw") is not None else None,
                "pitch": round(float(mp["pitch"]), 1) if mp.get("pitch") is not None else None,
                "gaze": mp.get("gaze"),
            })
        students = [
            {"name": name, "present": bool(state.get("is_present")), "ever_present": bool(state.get("ever_present"))}
            for name, state in sorted(self.proctoring_analyzer.student_presence_state.items())
        ]
        return {
            "classification": self.last_classification,
            "fps": round(fps, 1),
            "frames_processed": self.frames_processed,
            "heavy_model_updates": self.inference_count,
            "students": students,
            "people": people,
        }

    def get_status(self):
        with self._frame_cond:
            return dict(self._status)

    def wait_for_jpeg(self, after_seq=0, timeout=1.0):
        """Waits for a frame newer than after_seq.

        Returns (seq, jpeg bytes), or (after_seq, None) if no new frame arrived in time
        or the engine is stopping, so callers never resend the same frame.
        """
        with self._frame_cond:
            if self._jpeg_seq <= after_seq:
                self._frame_cond.wait_for(lambda: self._jpeg_seq > after_seq or self.stop_event.is_set(),
                                          timeout=timeout)
            if self._jpeg_seq > after_seq and self._jpeg is not None:
                return self._jpeg_seq, self._jpeg
            return after_seq, None

    def stop(self):
        """Asks run() to finish; safe to call from any thread."""
        self.stop_event.set()
        with self._frame_cond:
            self._frame_cond.notify_all()

    # --- Main Application Loop ---
    def run(self):
        self.logger.log_activity(f"Surveillance system started ({self.video_source.description}).")
        try:
            while not self.stop_event.is_set():
                frame = self.video_source.read()
                if frame is None:
                    if isinstance(self.video_source, FileSource):
                        self.logger.log_activity("Video file finished.")
                    else:
                        print("Error: Failed to grab frame. Exiting.")
                        self.logger.log_alert("CAMERA_ERROR", "Failed to grab frame.")
                    break

                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

                if self.display:
                    with self.processed_frame_lock:
                        shown = self.latest_processed_frame
                    cv2.imshow("Production-Ready Proctoring System", shown if shown is not None else frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.log_activity("User pressed 'q'. Shutting down.")
                        break
        finally:
            self.cleanup()

    # --- Cleanup (Unchanged, but includes previous bugfix) ---
    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        print("Shutting down system... signaling worker thread...")
        self.logger.log_core_debug("Cleanup sequence started. Signaling worker thread.") 
        self.stop()
        self.inference_stop.set()

        try:
            self.processing_thread.join(timeout=5.0)
        except Exception:
            try: self.processing_thread.join()
            except Exception: pass
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=10.0)

        print("Worker thread stopped. Releasing resources...")
        self.logger.log_core_debug("Worker thread stopped. Releasing resources.") 

        try:
            if getattr(self, "video_source", None) is not None:
                self.video_source.release()
                self.logger.log_core_debug("Video source released.")
        except Exception as e:
            print(f"[CLEANUP] Error releasing video source: {e}")
            self.logger.log_core_debug(f"Error releasing video source: {e}")

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

        if self.display:
            try:
                cv2.destroyAllWindows()
                self.logger.log_core_debug("OpenCV windows destroyed.")
            except Exception:
                pass
