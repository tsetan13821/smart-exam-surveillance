# app/utils/cpu_threads.py
"""CPU thread budget for the heavy models when they run beside the per-frame thread."""
import os

import config


def heavy_model_threads():
    """Threads RT-DETR / InsightFace may each use on CPU, or None for no cap.

    In async mode the heavy models run in a background thread while MediaPipe
    runs on every frame. PyTorch and ONNX Runtime both default to one thread per
    core, and with all three competing the per-frame thread falls behind and the
    heavy models slow down several-fold. A fixed share avoids that.
    """
    if not getattr(config, "HEAVY_MODELS_ASYNC", False):
        return None
    n = getattr(config, "CPU_HEAVY_MODEL_THREADS", None)
    if n is None:
        n = (os.cpu_count() or 4) // 4
    return max(1, int(n))
