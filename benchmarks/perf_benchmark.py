"""
Throughput benchmark for the full proctoring pipeline (ProctoringApp.process_frame).

Feeds frames from an image or video file, paced like a webcam (default 30 FPS),
through every stage (RT-DETR, InsightFace, MediaPipe, tracker, analyzers, drawing)
and reports, on this machine:

  analysed_fps          frames that went through MediaPipe + liveness + rules per second
                        (a webcam caps this at its own frame rate)
  max_gap_ms            longest time between two analysed frames; a blink shorter than
                        this can be missed entirely
  heavy_model_updates_per_s   how often RT-DETR + InsightFace results are refreshed
  per-stage time        mean ms per call and ms per actual inference

Works with both the background-inference mode and the older in-line mode.

Example:
  python benchmarks/perf_benchmark.py --source student.jpg --seconds 30 --out benchmarks/results/perf_after.json
"""
import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


class _QuietLogger:
    def __getattr__(self, name):
        return lambda *a, **k: None


class _NoCamera:
    def __init__(self, *a): pass
    def isOpened(self): return True
    def read(self): return True, np.zeros((480, 640, 3), np.uint8)
    def release(self): pass


def frame_source(path, n):
    img = cv2.imread(path)
    if img is not None:
        for _ in range(n):
            yield img.copy()
        return
    cap = cv2.VideoCapture(path)
    i = 0
    while i < n:
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        i += 1
        yield f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="image or video file used as camera input")
    ap.add_argument("--seconds", type=float, default=30.0, help="measured duration")
    ap.add_argument("--camera-fps", type=float, default=30.0, help="pace frames like a webcam")
    ap.add_argument("--warmup", type=float, default=5.0, help="seconds before measuring")
    ap.add_argument("--out")
    args = ap.parse_args()

    import app.core as core
    import torch
    from app.processors.face_recognizer import FaceRecognizer

    core.cv2.VideoCapture = _NoCamera
    logger = _QuietLogger()

    # Enroll whoever is in the first frame so recognition does real matching work.
    first = next(frame_source(args.source, 1))
    t = time.time()
    recognizer = FaceRecognizer(logger, {"Enrolled": [np.random.default_rng(0).normal(size=512)]})
    faces = recognizer.model.get(first)
    if faces:
        recognizer.known_encodings_normalized = [faces[0].embedding / np.linalg.norm(faces[0].embedding)]
    app = core.ProctoringApp(logger, recognizer, ["Enrolled"], exam_context={})
    app.stop_event.set()
    app.processing_thread.join()
    load_s = time.time() - t

    # Time each stage by wrapping it.
    stage_ms = defaultdict(list)

    def timed(name, fn):
        def wrapper(*a, **k):
            s = time.perf_counter()
            r = fn(*a, **k)
            stage_ms[name].append((time.perf_counter() - s) * 1000)
            return r
        return wrapper

    app.object_detector.detect = timed("RT-DETR (object detection)", app.object_detector.detect)
    app.face_processor.detect_faces = timed("InsightFace (face detect + identify)", app.face_processor.detect_faces)
    app.mediapipe_processor.process = timed("MediaPipe FaceMesh + pose/gaze", app.mediapipe_processor.process)
    app.person_tracker.update_tracks = timed("Tracker", app.person_tracker.update_tracks)
    app.proctoring_analyzer.analyze_frame = timed("Analyzers (liveness + rules)", app.proctoring_analyzer.analyze_frame)

    async_mode = getattr(app, "async_inference", False)
    period = 1.0 / args.camera_fps
    frames = frame_source(args.source, 10 ** 9)

    def run(seconds):
        """Feed frames at the camera rate for `seconds`; return per-frame ms and frame end times."""
        ms, ends = [], []
        start = next_due = time.perf_counter()
        while time.perf_counter() - start < seconds:
            now = time.perf_counter()
            if now < next_due:
                time.sleep(next_due - now)
            next_due = max(next_due + period, time.perf_counter() - period)   # drop frames if behind
            s = time.perf_counter()
            app.process_frame(next(frames))
            ms.append((time.perf_counter() - s) * 1000)
            ends.append(time.perf_counter())
        return ms, ends

    run(args.warmup)
    if async_mode:   # heavy models need a couple of results before the pipeline is in steady state
        deadline = time.perf_counter() + 120
        while app.inference_count < 2 and time.perf_counter() < deadline:
            run(0.5)
    for v in stage_ms.values():
        v.clear()

    inferences_at_start = getattr(app, "inference_count", 0)
    t_start = time.perf_counter()
    total, ends = run(args.seconds)
    measured_s = time.perf_counter() - t_start
    if async_mode:
        heavy_rate = (app.inference_count - inferences_at_start) / measured_s
        app.inference_stop.set()
        app.inference_thread.join(timeout=30)
    else:
        heavy_rate = sum(1 for v in stage_ms["RT-DETR (object detection)"] if v > 50) / measured_s
    gaps = np.diff([t_start] + ends) * 1000

    ort_providers = []
    try:
        ort_providers = sorted({p for m in recognizer.model.models.values() for p in m.session.get_providers()})
    except Exception:
        pass

    result = {
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "torch_device": app.object_detector.device,
                    "torch_cuda_available": torch.cuda.is_available(),
                    "onnxruntime_providers_in_use": ort_providers,
                    "insightface_models_loaded": sorted(getattr(recognizer.model, "models", {}).keys())},
        "heavy_models_async": async_mode,
        "camera_fps": args.camera_fps,
        "seconds_measured": round(measured_s, 1),
        "model_load_s": round(load_s, 1),
        "analysed_fps": round(len(total) / measured_s, 2),
        "max_gap_ms": round(float(gaps.max()), 1),
        "gap_ms_p95": round(float(np.percentile(gaps, 95)), 1),
        "heavy_model_updates_per_s": round(heavy_rate, 2),
        "frame_ms_mean": round(float(np.mean(total)), 1),
        "stages": {},
    }
    for name, v in stage_ms.items():
        v = np.array(v)
        # In in-line mode the heavy models run on every Nth call and return cached results otherwise
        heavy = v[v > 50] if name.startswith(("RT-DETR", "InsightFace")) else v
        result["stages"][name] = {
            "mean_ms_per_frame": round(float(v.mean()), 1),
            "ms_per_inference": round(float(heavy.mean() if len(heavy) else v.mean()), 1),
        }

    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
