# app/utils/liveness_helpers.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

"""
Provides helper functions for pixel-based liveness checks,
including optical flow and structural similarity (SSIM).
"""

def mean_optical_flow(prev_gray, curr_gray):
    """
    Calculates the mean magnitude of Farneback optical flow
    between two grayscale frames.
    """
    if prev_gray is None or curr_gray is None:
        return None
    
    # Ensure frames are the same size (e.g., if box size changed)
    if prev_gray.shape != curr_gray.shape:
        curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))

    if curr_gray is None: return None # Resize can fail
        
    try:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        if flow is None: return 0.0
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag))
    except Exception as e:
        print(f"Optical flow error: {e}")
        return 0.0 # Return 0 to avoid false negatives

def ssim_or_mse(prev_gray, curr_gray):
    """
    Calculates SSIM between two grayscale frames.
    Falls back to a normalized MSE calculation if SSIM fails.
    """
    if prev_gray is None or curr_gray is None:
        return None
        
    # Ensure frames are the same size
    if prev_gray.shape != curr_gray.shape:
        curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))
        
    if curr_gray is None: return 0.0

    try:
        # Calculate SSIM
        score = float(ssim(prev_gray, curr_gray))
        return score
    except Exception:
        # Fallback: normalized MSE -> 1.0 - normalized_mse
        try:
            mse = ((prev_gray.astype("float") - curr_gray.astype("float")) ** 2).mean()
            norm = 255.0 ** 2 # Max possible MSE
            if norm == 0: return 0.0
            
            # 1.0 = identical, 0.0 = max difference
            return max(0.0, 1.0 - (mse / norm))
        except Exception:
            return 0.0 # Return 0 (dissimilar) on failure