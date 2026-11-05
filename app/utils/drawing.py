import cv2
import mediapipe as mp
import numpy as np
import math

# --- REMOVED Landmark Contours ---
# Per your request, the eye and lip contour lists are removed.

def _conf_to_color(conf):
    """
    Map confidence [0,1] -> BGR color:
      low -> red, medium -> orange, high -> green
    """
    if conf is None:
        return (0, 255, 255)  # default yellow
    # piecewise simple ramp
    if conf < 0.33:
        return (0, 0, 255)    # red
    elif conf < 0.66:
        return (0, 165, 255)  # orange
    else:
        return (0, 200, 0)    # green

def draw_detections(frame, persons):
    """Draws bounding boxes and labels for detected persons."""
    for person in persons:
        x1, y1, x2, y2 = person['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

def draw_contraband(frame, items):
    """Draws bounding boxes and labels for detected contraband."""
    for item in items:
        x1, y1, x2, y2 = item['box']
        
        # --- NEW: Add confidence score to label ---
        label = f"{item['label']} ({item['score']:.2f})"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)


def draw_face_annotations(frame, tracked_persons):
    """
    Draws advanced face annotations, including:
    1. Bounding Box with Identity & Spoof color-coding.
    2. A prominent "ALERT" banner if a cheat is detected.
    3. Live Gaze & Head Pose text for debugging.
    4. --- MODIFIED ---
       Draws a semi-transparent cone that turns RED when looking away.
    """
    
    if not tracked_persons:
        return

    # --- NEW: Create a single overlay for all transparencies ---
    overlay = frame.copy()
    
    # --- NEW: Define transparency for the gaze cone ---
    ALPHA = 0.2

    # Iterate through each tracked person
    for person in tracked_persons.values():
        try:
            x1, y1, x2, y2 = person.box
            label = person.get_display_name()
            
            # --- 1. Determine Color (Identity & Spoof) ---
            if person.is_spoofing:
                color = (0, 165, 255) # Orange for spoof
            elif person.name == "Unknown":
                color = (0, 0, 255) # Red for unknown
            else:
                color = (0, 255, 0) # Green for student

            # --- 2. Draw Main Bounding Box & Label ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_y_pos = y1 - 10 

            # --- 3. NEW: Draw Alert Banner (Testing Feature) ---
            alert_text = ""
            if person.is_spoofing:
                alert_text = "IMAGE DETECTED"
            # Check for analyzer alerts (e.g., "HEAD DOWN")
            elif hasattr(person, 'current_alerts') and person.current_alerts:
                alert_text = person.current_alerts[0].upper() 
            
            if alert_text:
                color = (0, 0, 255) # Use Red as the base color
                cv2.rectangle(frame, (x1, y1 - 28), (x2, y1), color, -1)
                cv2.putText(frame, alert_text, (x1 + 5, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                label_y_pos = y1 - 35 # Move label up to clear banner

            cv2.putText(frame, label, (x1, label_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # --- 4. Landmark-dependent drawings ---
            if person.mp_result:
                # --- SECTION 4: Draw Gaze & Pose Text ---
                gaze = person.mp_result.get("gaze", "...")
                yaw = float(person.mp_result.get("yaw", 0.0) or 0.0)
                pitch = float(person.mp_result.get("pitch", 0.0) or 0.0)
                conf = person.mp_result.get("confidence", None)

                # --- NEW "PRACTICAL" FEATURE ---
                # Determine the color for gaze text AND the cone
                if gaze not in ("Center", "Uncertain", "..."):
                    gaze_alert_color = (0, 0, 255) # Red
                else:
                    gaze_alert_color = (255, 255, 255) # White

                gaze_text = f"GAZE: {gaze}"
                pose_text = f"POSE: YAW({yaw:.0f}), PITCH({pitch:.0f})"
                conf_text = f"CONF: {conf:.2f}" if isinstance(conf, (int,float)) else "CONF: --"

                cv2.putText(frame, gaze_text, (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, gaze_alert_color, 2)
                cv2.putText(frame, pose_text, (x1, y2 + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, conf_text, (x1, y2 + 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _conf_to_color(conf), 2)
                
                # --- SECTION 5: Draw Dynamic Gaze Cone ---
                
                gaze_proj = person.mp_result.get("gaze_proj_2d")
                left_pupil = person.mp_result.get("left_pupil_2d")
                right_pupil = person.mp_result.get("right_pupil_2d")

                # We must have the projected hit point to draw the cone
                if gaze_proj and left_pupil and right_pupil:
                    
                    # --- NEW: Set cone color ---
                    # If gaze is "Center", use confidence color.
                    # If gaze is "Gaze Left"/"Gaze Right", use RED.
                    if gaze_alert_color == (0, 0, 255): # Red
                        cone_color = (0, 0, 255) # BGR Red
                    else:
                        cone_color = _conf_to_color(conf) # Default

                    # The single target point
                    p_hit = (int(gaze_proj[0]), int(gaze_proj[1]))
                    
                    # The origin points (the pupils)
                    p1_l = (int(left_pupil[0]), int(left_pupil[1]))
                    p1_r = (int(right_pupil[0]), int(right_pupil[1]))
                            
                    # --- Define cone vertices ---
                    GAZE_RADIUS = 150 # Radius of the gaze cone at the hit point
                    p_hit_left = (p_hit[0] - GAZE_RADIUS, p_hit[1])
                    p_hit_right = (p_hit[0] + GAZE_RADIUS, p_hit[1])

                    # Create a polygon from pupils to the gaze area
                    points = np.array([p1_l, p1_r, p_hit_right, p_hit_left], dtype=np.int32)
                    
                    # --- Draw the transparent polygon on the overlay ---
                    cv2.fillPoly(overlay, [points], cone_color, cv2.LINE_AA)
        
        except Exception as e:
            print(f"Error drawing annotations for person {person.id}: {e}")
            continue # Move to the next person

    # --- Blend the overlay with the original frame ---
    # This must be done *after* the loop, so all cones are drawn
    cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)


def draw_status_panel(frame, classification):
    """Draws the final classification status on the top-left of the frame."""
    if classification == "Normal":
        color = (0, 255, 0) # Green
    else:
        color = (0, 0, 255) # Red

    cv2.putText(frame, classification, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)