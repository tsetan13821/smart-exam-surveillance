# pages/1_📝_Register.py
import streamlit as st
import cv2
import numpy as np
import json
import os
import uuid
import time
import av
import traceback
import queue  # <-- Re-added for thread-safe rerun
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- Import from core files ---
from src.config import (
    REGISTRATION_EVIDENCE_DIR,
    POSE_REQUIREMENTS, DEBOUNCE_FRAMES
)
from src.database import get_registration, save_registration
from src.models import get_models
from src.pose_analysis import calculate_simple_head_pose

# ==================================================================
# ---           REGISTRATION VIDEO TRANSFORMER (Refined)         ---
# ==================================================================
class RegistrationTransformer(VideoTransformerBase):
    # Add completion_queue to __init__
    def __init__(self, student_id: str, student_name: str, completion_queue: queue.Queue):
        print("Initializing RegistrationTransformer...")
        self.student_id = student_id
        self.student_name = student_name
        self.completion_queue = completion_queue  # <-- Re-added
        
        self.arcface_model, self.face_mesh_model = get_models()
        
        if not self.arcface_model or not self.face_mesh_model:
            self.state = "ERROR"
            self.error_message = "FATAL: AI Models failed to load."
            return
            
        self.state = "INITIALIZING"
        self.message = "Initializing..."
        self.message_color = (0, 255, 255)
        self.poses_to_capture = ["front", "left", "right"]
        self.pose_debounce_counter = 0
        self.captured_images = {}
        self.captured_encodings = {}
        self.temp_message = None
        self.temp_message_time = 0
        self.review_evidence_id = str(uuid.uuid4())
        self.error_message = None
        self.last_yaw = 0.0
        self.current_target_range = (0, 0)
        print(f"RegistrationTransformer initialized for {self.student_id}.")

    def _process_and_save(self):
        """Processes all captured images, generates encodings, and saves to DB."""
        try:
            print("Processing encodings for all captured images...")
            self.message = "Processing... This may take a moment."
            self.message_color = (0, 255, 255)
            
            student_id = self.student_id
            student_name = self.student_name

            if not student_id or not student_name:
                self.state = "ERROR"
                self.error_message = "ERROR: Student ID/Name invalid. Please refresh."
                return

            for pose_name, img_bgr in self.captured_images.items():
                evidence_path = os.path.join(
                    REGISTRATION_EVIDENCE_DIR, 
                    f"{self.review_evidence_id}_{pose_name}.jpg"
                )
                cv2.imwrite(evidence_path, img_bgr)
                print(f"Saved {pose_name} evidence to {evidence_path}")

            print(f"Saving registration for {student_id}...")
            save_registration(student_id, student_name, list(self.captured_encodings.values()),
                              self.review_evidence_id)
            
            print("Registration complete!")
            self.state = "COMPLETE"
            st.session_state.registration_complete = True
            
            # --- FIXED: Use queue instead of st.rerun() ---
            print("Putting 'DONE' signal into completion queue.")
            self.completion_queue.put("DONE")
            # --- END OF FIX ---
            
        except Exception as e:
            self.state = "ERROR"; self.error_message = f"Processing Error: {e}"
            print(f"FATAL processing error: {e}")
            traceback.print_exc()
            self.completion_queue.put("ERROR") # Signal error

    # --- _draw_guidance_bar (Unchanged) ---
    def _draw_guidance_bar(self, img_bgr, h, w):
        bar_y, bar_height, bar_width = h - 65, 20, 400
        max_display_yaw = 60.0
        center_x = w // 2
        bar_start_x = center_x - (bar_width // 2)
        bar_end_x = center_x + (bar_width // 2)
        pixels_per_degree = bar_width / (max_display_yaw * 2)
        cv2.rectangle(img_bgr, (bar_start_x, bar_y), (bar_end_x, bar_y + bar_height), (100, 100, 100), -1)
        min_yaw, max_yaw = self.current_target_range
        target_start_x = int(center_x + (min_yaw * pixels_per_degree))
        target_end_x = int(center_x + (max_yaw * pixels_per_degree))
        target_start_x = max(bar_start_x, target_start_x)
        target_end_x = min(bar_end_x, target_end_x)
        cv2.rectangle(img_bgr, (target_start_x, bar_y), (target_end_x, bar_y + bar_height), (0, 255, 0), -1)
        current_yaw_x = int(center_x + (self.last_yaw * pixels_per_degree))
        current_yaw_x = max(bar_start_x, min(bar_end_x, current_yaw_x))
        cv2.line(img_bgr, (current_yaw_x, bar_y - 5), (current_yaw_x, bar_y + bar_height + 5), (255, 255, 255), 3)
        cv2.putText(img_bgr, "Right", (bar_start_x - 50, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(img_bgr, "Front", (center_x - 25, bar_y + bar_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(img_bgr, "Left", (bar_end_x + 10, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # --- recv (Refactored) ---
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr_original = frame.to_ndarray(format="bgr24")
        img_bgr_display = cv2.flip(img_bgr_original, 1) # 1 = horizontal flip
        h, w, _ = img_bgr_display.shape

        # Handle final states
        if self.state == "COMPLETE": # <-- Re-added COMPLETE state
            self.message = "Registration Submitted!"
            self.message_color = (0, 255, 0)
        elif self.state == "PROCESSING":
            self.message = "Processing... This may take a moment."
            self.message_color = (0, 255, 255)
        elif self.state == "ERROR":
            self.message = self.error_message
            self.message_color = (0, 0, 255)
        
        # --- Main State: CAPTURING ---
        elif self.state == "INITIALIZING" or self.state == "CAPTURING":
            self.state = "CAPTURING"
            
            if not self.poses_to_capture:
                self.state = "PROCESSING"
                self._process_and_save()
            else:
                current_pose_name = self.poses_to_capture[0]
                req = POSE_REQUIREMENTS[current_pose_name]
                min_yaw, max_yaw = req["range"]
                self.current_target_range = req["range"]
                
                img_rgb = cv2.cvtColor(img_bgr_original, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                results = self.face_mesh_model.process(img_rgb)
                yaw, pitch = calculate_simple_head_pose(results, h, w)
                
                if self.temp_message and time.time() - self.temp_message_time > 3:
                    self.temp_message = None

                if yaw is None:
                    self.message = "No face detected or poor lighting."
                    self.message_color = (0, 0, 255)
                    self.pose_debounce_counter = 0
                    self.last_yaw = 0.0
                else:
                    self.last_yaw = yaw
                    
                    if min_yaw <= yaw <= max_yaw:
                        self.pose_debounce_counter += 1
                        self.message_color = (0, 255, 0)
                        self.message = f"Hold {current_pose_name} pose... {self.pose_debounce_counter}/{DEBOUNCE_FRAMES}"
                        
                        if self.pose_debounce_counter >= DEBOUNCE_FRAMES:
                            
                            print(f"Pose held. Running ArcFace check for {current_pose_name}...")
                            faces = self.arcface_model.get(img_bgr_original)
                            
                            if not faces:
                                print("ArcFace: No face found.")
                                self.temp_message = "No face found, please try again."
                                self.temp_message_time = time.time()
                                self.pose_debounce_counter = 0
                            elif len(faces) > 1:
                                print(f"ArcFace: Multiple faces ({len(faces)}) found.")
                                self.temp_message = "Multiple faces detected, please try again."
                                self.temp_message_time = time.time()
                                self.pose_debounce_counter = 0
                            else:
                                # --- NEW: Closeness Check ---
                                face = faces[0]
                                bbox = face.bbox.astype(int)
                                x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
                                face_area = (x2 - x1) * (y2 - y1)
                                frame_area = h * w
                                size_ratio = face_area / frame_area
                                
                                if size_ratio < 0.10: # Face < 10% of frame
                                    print(f"ArcFace: Face too small ({size_ratio:.2f}).")
                                    self.temp_message = "Too far away. Please move closer."
                                    self.temp_message_time = time.time()
                                    self.pose_debounce_counter = 0
                                elif size_ratio > 0.40: # Face > 40% of frame
                                    print(f"ArcFace: Face too large ({size_ratio:.2f}).")
                                    self.temp_message = "Too close. Please move back."
                                    self.temp_message_time = time.time()
                                    self.pose_debounce_counter = 0
                                else:
                                    # --- SUCCESS: All checks passed! ---
                                    print(f"ArcFace & Size OK. Capturing {current_pose_name} pose!")
                                    self.captured_images[current_pose_name] = img_bgr_original.copy()
                                    self.captured_encodings[current_pose_name] = face.embedding
                                    self.poses_to_capture.pop(0)
                                    self.pose_debounce_counter = 0
                                    self.temp_message = None
                                    time.sleep(0.5) # Pause
                                # --- END NEW ---
                    else:
                        self.pose_debounce_counter = 0
                        self.message_color = (0, 255, 255)
                        base_message = req["message"]
                        
                        if current_pose_name == "front":
                            self.message = f"{base_message} (Turn slightly {'RIGHT' if yaw > max_yaw else 'LEFT'})"
                        elif current_pose_name == "left":
                            self.message = f"{base_message} (Turn {'MORE to your left' if yaw < min_yaw else 'back to your RIGHT'})"
                        elif current_pose_name == "right":
                            self.message = f"{base_message} (Turn {'MORE to your right' if yaw > max_yaw else 'back to your LEFT'})"
                
                if self.temp_message:
                    self.message = self.temp_message
                    self.message_color = (0, 0, 255)
        
        cv2.putText(img_bgr_display, self.message, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.message_color, 2)
        
        if self.state == "CAPTURING" and self.poses_to_capture:
            self._draw_guidance_bar(img_bgr_display, h, w)

        return av.VideoFrame.from_ndarray(img_bgr_display, format="bgr24")

# ==================================================================
# ---                  STREAMLIT APP UI (Updated)                ---
# ==================================================================
st.title("Step 1: 📝 Biometric Registration")

models_loaded = get_models()
if not all(models_loaded):
    st.error("Fatal Error: AI Models could not be loaded. Please check the console logs and restart.")
    st.stop()

st.subheader("Enter Your Information")
col1, col2 = st.columns(2)
with col1:
    student_id = st.text_input("Student ID (e.g., s1001)", key="student_id")
with col2:
    student_name = st.text_input("Full Name", key="student_name")

st.divider()

if student_id and student_name:
    
    student_record = get_registration(st.session_state.student_id)
    status = student_record.get("status") if student_record else None

    # --- Re-added Access Control ---
    if status == "approved":
        st.success("✅ Your registration is already **approved**. You do not need to register again.")
        st.stop() # Stop the page
        
    elif status == "pending":
        st.warning("⏳ Your registration is **pending review**. You do not need to register again.")
        st.stop() # Stop the page

    elif st.session_state.get('registration_complete', False):
        st.success("🎉 Registration Submitted Successfully! It is now pending review.")
        st.balloons()
        st.stop() # Stop the page

    elif status is None or status == "rejected":
        # --- GOOD TO PROCEED ---
        if status == "rejected":
            st.error("Your previous submission was rejected. Please try again, following the instructions carefully.")
        
        st.info("Follow the on-screen instructions. Use the visual bar at the bottom to help guide your head position.")
        
        # --- Re-added Queue Logic ---
        completion_queue = queue.Queue()

        webrtc_streamer(
            key="registration-auto",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: RegistrationTransformer(
                student_id=student_id,
                student_name=student_name,
                completion_queue=completion_queue # Pass the queue in
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Main thread polling loop
        while True:
            try:
                result = completion_queue.get(timeout=0.1)
                if result == "DONE":
                    print("Main thread received 'DONE' signal. Rerunning.")
                    st.rerun() # This is now SAFE
                    break
                elif result == "ERROR":
                     print("Main thread received 'ERROR' signal. Stopping.")
                     break
            except queue.Empty:
                time.sleep(0.1)
                pass
    
    else:
        st.error(f"Your account has an unknown status ('{status}'). Please contact an administrator.")
        st.stop()
    # --- END OF Re-added Logic ---

else:
    st.info("Please enter your ID and Name to begin registration.")