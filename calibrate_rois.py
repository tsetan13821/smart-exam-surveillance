# calibrate_rois.py
import cv2
import json
import numpy as np
import sys
import os

# Attempt to import config to get the camera index
try:
    import config as config
    CAMERA_INDEX = getattr(config, "CAMERA_INDEX", 0)
except ImportError:
    print("WARNING: Could not import config.py. Defaulting to camera index 0.")
    CAMERA_INDEX = 0

# --- Global Variables for Mouse Callback ---
polygons = {}           # Dict to store {name: [[x,y], ...]}
current_polygon = []    # List of (x,y) points for the polygon being drawn
current_name = ""       # Name of the student/desk for the current polygon
drawing = False         # True if the user is actively drawing a polygon
frame_copy = None       # A clean copy of the frame to re-draw on

# --- Constants ---
ROI_FILE = "desk_rois.json"
WINDOW_NAME = "Desk ROI Calibration Tool"

def load_rois():
    """Loads existing ROIs from the JSON file."""
    global polygons
    if not os.path.exists(ROI_FILE):
        print(f"No existing {ROI_FILE} found. Starting fresh.")
        polygons = {}
        return
        
    try:
        with open(ROI_FILE, 'r') as f:
            polygons = json.load(f)
        print(f"Loaded {len(polygons)} existing ROIs from {ROI_FILE}")
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse {ROI_FILE}. Starting fresh.")
        polygons = {}
    except Exception as e:
        print(f"An unknown error occurred loading {ROI_FILE}: {e}. Starting fresh.")
        polygons = {}


def save_rois():
    """Saves the current ROIs to the JSON file."""
    try:
        with open(ROI_FILE, 'w') as f:
            json.dump(polygons, f, indent=4)
        print(f"Successfully saved {len(polygons)} ROIs to {ROI_FILE}")
    except IOError as e:
        print(f"FATAL: Could not save ROIs to {ROI_FILE}. Error: {e}")

def draw_instructions(frame):
    """Draws help text on the frame."""
    h, w, _ = frame.shape
    instructions = [
        "[n] = New Desk (Enter name in console)",
        "[LEFT-CLICK] = Add point",
        "[RIGHT-CLICK] = Finish & Save Polygon",
        "[c] = Clear last point",
        "[r] = Reset ALL ROIs",
        "[s] = Save & Quit",
        "[q] = Quit WITHOUT saving"
    ]
    
    y = 30
    for text in instructions:
        # Draw white text with a black outline for readability
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 30
    
    if drawing and current_name:
        status_text = f"Drawing for: {current_name}"
        status_color = (0, 0, 255) # Red
    elif current_name:
         status_text = f"Click to edit points for {current_name}"
         status_color = (0, 255, 0) # Green
    else:
        status_text = "Press 'n' to start a new desk."
        status_color = (0, 255, 255) # Yellow

    cv2.putText(frame, status_text, (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, status_text, (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 1)


def draw_all_polygons(frame):
    """Draws all saved polygons and the current in-progress one."""
    
    # Draw saved polygons
    for name, points in polygons.items():
        if points:
            pts_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_np], isClosed=True, color=(0, 255, 0), thickness=2)
            # Put the name at the first point
            cv2.putText(frame, name, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    # Draw current polygon
    if drawing and current_polygon:
        pts_np = np.array(current_polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_np], isClosed=False, color=(0, 0, 255), thickness=2)
        for point in current_polygon:
            cv2.circle(frame, tuple(point), 4, (0, 0, 255), -1)

def mouse_callback(event, x, y, flags, param):
    """Handles mouse events for drawing polygons."""
    global current_polygon, drawing, frame_copy
    
    if not drawing:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append([x, y])
        print(f"Added point: {(x, y)}")
        frame_copy = None # Force a redraw

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_polygon) < 3:
            print("ERROR: A polygon must have at least 3 points.")
            return
        
        print(f"Finished polygon for {current_name} with {len(current_polygon)} points.")
        polygons[current_name] = current_polygon
        
        # Reset for next polygon
        current_polygon = []
        drawing = False
        save_rois() # Auto-save on completion
        frame_copy = None # Force a redraw of the clean frame

def main():
    global drawing, current_name, current_polygon, polygons, frame_copy

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"FATAL: Cannot open camera at index {CAMERA_INDEX}.")
        sys.exit(1)
        
    load_rois()
    
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    
    print("\n--- Desk ROI Calibration Tool ---")
    print("Press 'n' in the window to start defining a new desk.")
    print("Follow the on-screen instructions.")
    print("IMPORTANT: The name you enter MUST MATCH the student's name in the registry.")

    while True:
        if frame_copy is None:
            # Only grab a new frame if we need to redraw
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()
        
        display_frame = frame_copy.copy()
        
        draw_all_polygons(display_frame)
        draw_instructions(display_frame)
        
        cv2.imshow(WINDOW_NAME, display_frame)
        key = cv2.waitKey(20) & 0xFF # Added a 20ms wait

        if key == ord('q'):
            print("Quitting without saving last changes...")
            break
        
        elif key == ord('s'):
            print("Saving and quitting...")
            save_rois()
            break
            
        elif key == ord('n'):
            name_input = input("\n>>> Enter the Student Name for this desk (must match registry): ")
            if name_input:
                if name_input in polygons:
                    print(f"WARNING: '{name_input}' already exists. Overwriting.")
                current_name = name_input
                current_polygon = []
                drawing = True
                frame_copy = None # Force refresh
                print(f"Started drawing for '{current_name}'. Click to add points, right-click to finish.")
            else:
                print("No name entered. Canceled.")
                
        elif key == ord('c'):
            if drawing and current_polygon:
                removed_point = current_polygon.pop()
                print(f"Removed last point: {removed_point}")
                frame_copy = None # Force refresh
            else:
                print("Not drawing, cannot remove point.")
            
        elif key == ord('r'):
            confirm = input("\n>>> ARE YOU SURE you want to clear ALL ROIs? (y/n): ")
            if confirm.lower() == 'y':
                polygons = {}
                current_polygon = []
                drawing = False
                save_rois()
                frame_copy = None # Force refresh
                print("All ROIs have been reset.")
        
        # If not drawing, we don't need to refresh the frame constantly
        if not drawing:
            frame_copy = None


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()