# app/utils/logger.py
import cv2
import json
import os
import sqlite3
import config as config
from datetime import datetime
import queue
import threading

class ActivityLogger:
    """
    Handles logging to both a .jsonl file and the SQLite database.
    
    --- MODIFICATION (v2) ---
    This class is now asynchronous AND supports multi-student sessions.
    The constructor accepts JSON lists of student IDs/names.
    """
    
    # --- MODIFIED: Updated constructor signature ---
    def __init__(self, session_id, student_ids_json, student_names_json, log_dir="proctoring_logs"):
        self.session_id = session_id
        # --- MODIFIED: Store the JSON strings ---
        self.student_ids_json = student_ids_json
        self.student_names_json = student_names_json
        
        self.log_dir = log_dir
        self.evidence_dir = os.path.join(self.log_dir, "evidence")

        self.log_file_path = os.path.join(self.log_dir, f"session_{self.session_id}.log")
        self.action_log_dir = config.HUMAN_ACTION_LOG_DIR
        self.action_log_file_path = os.path.join(self.action_log_dir, f"session_{self.session_id}_actions.log")
        self.core_log_dir = config.CORE_LOGS_DIR
        self.core_log_file_path = os.path.join(self.core_log_dir, f"session_{self.session_id}_core.log")
        self.liveness_log_dir = config.LIVENESS_LOGS_DIR
        self.liveness_log_file_path = os.path.join(self.liveness_log_dir, f"session_{self.session_id}_liveness.log")
        self.similarity_log_dir = config.SIMILARITY_LOGS_DIR
        self.similarity_log_file_path = os.path.join(self.similarity_log_dir, f"session_{self.session_id}_similarity.log")
        self.mediapipe_log_dir = config.MEDIAPIPE_LOGS_DIR
        self.mediapipe_log_file_path = os.path.join(self.mediapipe_log_dir, f"session_{self.session_id}_mediapipe.log")
        self.object_detector_log_dir = config.OBJECT_DETECTOR_LOGS_DIR
        self.object_detector_log_file_path = os.path.join(self.object_detector_log_dir, f"session_{self.session_id}_detection.log")
        self.db_path = config.DATABASE_FILE

        self._ensure_dirs_exist()

        # --- MODIFIED: Updated print statement ---
        print(f"Logging session {self.session_id} (Multi-Student)")
        print(f"  .jsonl Log: {self.log_file_path}")
        print(f"  Database: {self.db_path}")
        
        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()
        print("Asynchronous logger worker thread started.")

    def _ensure_dirs_exist(self):
        """Creates the log and evidence directories if they don't exist."""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.evidence_dir, exist_ok=True)
        os.makedirs(self.action_log_dir, exist_ok=True)
        os.makedirs(self.core_log_dir, exist_ok=True)
        os.makedirs(self.liveness_log_dir, exist_ok=True)
        os.makedirs(self.similarity_log_dir, exist_ok=True)
        os.makedirs(self.mediapipe_log_dir, exist_ok=True) 
        os.makedirs(self.object_detector_log_dir, exist_ok=True)

    def _log_worker(self):
        """
        The dedicated worker thread function.
        Continuously pulls tasks from self.log_queue and executes them.
        """
        while not self.stop_event.is_set() or not self.log_queue.empty():
            try:
                action, payload = self.log_queue.get(timeout=1.0)
                
                if action == "_stop":
                    break
                elif action == "_write_log_jsonl":
                    self._write_log_jsonl(payload)
                elif action == "_write_log_db":
                    self._write_log_db(*payload)
                elif action == "_write_action_log":
                    self._write_action_log(payload)
                elif action == "_write_debug_log":
                    self._write_debug_log(*payload)
                
                self.log_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"LOGGER_WORKER_ERROR: {e}")
                
    def _write_debug_log(self, file_path, message):
        """
        A unified helper for writing simple, timestamped debug messages
        to a specified file.
        """
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(message + "\n")
        except Exception as e:
            print(f"LOGGER_WRITE_FAILURE: {e}")
            print(f"ORIGINAL_MSG: {message}")

    # --- I/O Methods (Called by worker) ---

    def _write_log_jsonl(self, log_entry):
        try:
            # --- MODIFIED: Add the JSON lists to the log entry ---
            log_entry['session_id'] = self.session_id
            log_entry['student_ids'] = self.student_ids_json
            log_entry['student_names'] = self.student_names_json
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                json_string = json.dumps(log_entry)
                f.write(json_string + '\n')
        except Exception as e:
            print(f"LOGGER WARNING: Could not write to .jsonl file: {e}")

    def _write_action_log(self, log_entry):
        try:
            # --- MODIFIED: Add session/student info ---
            log_entry['session_id'] = self.session_id
            log_entry['student_ids'] = self.student_ids_json
            with open(self.action_log_file_path, 'a', encoding='utf-8') as f:
                json_string = json.dumps(log_entry)
                f.write(json_string + '\n')
        except Exception as e:
            print(f"LOGGER WARNING: Could not write to action log file: {e}")

    def _write_log_db(self, level, message, alert_type=None, evidence_file=None):
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO alerts (session_id, timestamp, level, alert_type, message, evidence_file)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.session_id,
                        datetime.now().isoformat(),
                        level,
                        alert_type,
                        message,
                        evidence_file
                    )
                )
                conn.commit()
        except sqlite3.Error as e:
            print(f"LOGGER FATAL: Could not write to database: {e}")

    # --- PUBLIC LOGGING METHODS (ASYNC) ---
    def log_detection_debug(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        formatted_message = f"[{timestamp}] {message}"
        self.log_queue.put( ("_write_debug_log", (self.object_detector_log_file_path, formatted_message)) )
    
    def log_activity(self, message):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": message
        }
        self.log_queue.put( ("_write_log_jsonl", log_entry) )
        self.log_queue.put( ("_write_log_db", ("INFO", message, "ACTIVITY", None)) )

    def log_alert(self, alert_type, message):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "ALERT",
            "alert_type": alert_type,
            "message": message
        }
        self.log_queue.put( ("_write_log_jsonl", log_entry) )
        self.log_queue.put( ("_write_log_db", ("ALERT", message, alert_type, None)) )

    def log_human_action(self, action_key, action_value, details_dict=None):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action_key": action_key,
            "action_value": action_value,
            "details": details_dict or {}
        }
        self.log_queue.put( ("_write_action_log", log_entry) )

    def log_evidence(self, frame, alert_type, message):
        """
        Saves an evidence frame.
        """
        try:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            
            # --- MODIFIED: Filename no longer uses student_id ---
            # It's ambiguous in a multi-student session.
            # The file is stored in the session-specific folder anyway.
            filename = f"{alert_type}_{timestamp_str}.jpg"
            
            evidence_path = os.path.join(self.evidence_dir, filename)
            evidence_abs_path = os.path.abspath(evidence_path)
            
            cv2.imwrite(evidence_path, frame)
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "EVIDENCE",
                "alert_type": alert_type,
                "message": message,
                "evidence_file": evidence_abs_path
            }
            
            self.log_queue.put( ("_write_log_jsonl", log_entry) )
            self.log_queue.put( (
                "_write_log_db",
                ("EVIDENCE", message, alert_type, evidence_abs_path)
            ) )
            return evidence_abs_path
            
        except Exception as e:
            print(f"ERROR: Failed to save evidence for {alert_type}: {e}")
            self.log_alert("EVIDENCE_FAILURE", f"Failed to save evidence for {alert_type}: {str(e)}")
            return None

    def log_core_debug(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        formatted_message = f"[{timestamp}] {message}"
        self.log_queue.put( ("_write_debug_log", (self.core_log_file_path, formatted_message)) )

    def log_liveness_debug(self, message, person_name=None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        name_str = f" [{person_name}]" if person_name else ""
        formatted_message = f"[{timestamp}]{name_str} {message}"
        self.log_queue.put( ("_write_debug_log", (self.liveness_log_file_path, formatted_message)) )

    def log_similarity_debug(self, message, person_name=None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        name_str = f" [{person_name}]" if person_name else ""
        formatted_message = f"[{timestamp}]{name_str} {message}"
        self.log_queue.put( ("_write_debug_log", (self.similarity_log_file_path, formatted_message)) )

    def log_mediapipe_debug(self, event_type, message, details=None, person_name=None):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "person_name": person_name,
            "event_type": event_type,
            "message": message,
            "details": details or {}
        }
        json_string = json.dumps(log_entry)
        self.log_queue.put( ("_write_debug_log", (self.mediapipe_log_file_path, json_string)) )

    def close(self):
        """
        Signals the logger thread to shut down and waits for it to finish.
        """
        print("Logger close() called. Signaling worker thread...")
        self.log_activity("Logger shutting down.")
        
        self.stop_event.set()
        self.log_queue.put( ("_stop", None) )
        
        self.log_thread.join(timeout=5.0)
        print("Logger worker thread has shut down.")