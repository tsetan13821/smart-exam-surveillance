# core/pose_analysis.py
import cv2
import numpy as np

def _get_landmarks_from_mesh(face_landmarks, indices, h, w):
    coords = []
    try:
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            coords.append((lm.x * w, lm.y * h))
        return coords
    except Exception:
        return None

def calculate_simple_head_pose(face_mesh_results, h, w):
    if not face_mesh_results.multi_face_landmarks:
        return None, None
    
    face_landmarks = face_mesh_results.multi_face_landmarks[0]
    HEAD_POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]
    
    face_2d = _get_landmarks_from_mesh(face_landmarks, HEAD_POSE_LANDMARKS, h, w)
    if face_2d is None: return None, None
    
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array([
        [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
    ], dtype=np.float64)
    
    try:
        cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)
        if not success: return None, None
        
        rmat, _ = cv2.Rodrigues(rot_vec)
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
            yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        else:
            pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
            yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        
        return yaw, pitch
    except Exception:
        return None, None