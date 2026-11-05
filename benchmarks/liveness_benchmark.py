"""
Liveness / anti-spoofing benchmark.

Runs the real MediaPipeProcessor + LivenessDetector + PersonAnalyzer over video
clips and reports how often spoofs are caught and how often real people are
wrongly flagged.

Two kinds of input:

  --images DIR   Photos of faces (one face per photo). Each is turned into
                 synthetic SPOOF clips that imitate a photo held up to a webcam:
                 two face sizes x three camera-noise levels x
                 {held still, hand-held shake, tilting back and forth}.

  --clips DIR    Real recordings in DIR/live/*.mp4 and DIR/spoof/*.mp4
                 (record them with benchmarks/record_clip.py).

Time is simulated at the clip's frame rate, so results don't depend on CPU speed.

Example:
  python benchmarks/liveness_benchmark.py --images path/to/faces --clips benchmarks/clips --out benchmarks/results/liveness_after.json
"""
import argparse
import datetime as _dt
import glob
import importlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

FRAME_W, FRAME_H = 640, 480


# --------------------------------------------------------------------------- clock
class _Clock:
    t = _dt.datetime(2026, 1, 1, 9, 0, 0)


class _FakeDateTime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return _Clock.t


def _patch_clock():
    for m in ("app.processors.liveness_detector", "app.processors.person_analyzer",
              "app.models.tracked_person"):
        importlib.import_module(m).datetime = _FakeDateTime


class _QuietLogger:
    def __init__(self):
        self.evidence = []

    def log_evidence(self, frame, alert_type, message):
        self.evidence.append(alert_type)

    def __getattr__(self, name):
        return lambda *a, **k: None


# --------------------------------------------------------------- synthetic spoofs
def _face_box(img_bgr):
    """Face bounding box via MediaPipe FaceMesh in static-image mode."""
    import mediapipe as mp
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        res = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    h, w = img_bgr.shape[:2]
    pts = np.array([(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark])
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return x1, y1, x2, y2


def _place_face(img_bgr, box, face_px):
    """Webcam-like 640x480 frame with the face scaled to `face_px` wide, centred."""
    x1, y1, x2, y2 = box
    s = face_px / max(1.0, (x2 - x1))
    big = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    cx, cy = (x1 + x2) / 2 * s, (y1 + y2) / 2 * s
    frame = np.full((FRAME_H, FRAME_W, 3), 90, np.uint8)
    ox, oy = int(FRAME_W / 2 - cx), int(FRAME_H * 0.45 - cy)
    sx1, sy1 = max(0, -ox), max(0, -oy)
    dx1, dy1 = max(0, ox), max(0, oy)
    w = min(big.shape[1] - sx1, FRAME_W - dx1)
    h = min(big.shape[0] - sy1, FRAME_H - dy1)
    if w > 0 and h > 0:
        frame[dy1:dy1 + h, dx1:dx1 + w] = big[sy1:sy1 + h, sx1:sx1 + w]
    return frame


def _handheld(frame, i, amp=3.0):
    """Small translation + rotation, like a phone held in a hand."""
    M = cv2.getRotationMatrix2D((FRAME_W / 2, FRAME_H / 2), amp * 0.5 * np.sin(i / 4), 1.0)
    M[0, 2] += amp * np.sin(i / 3)
    M[1, 2] += amp * np.cos(i / 5)
    return cv2.warpAffine(frame, M, (FRAME_W, FRAME_H), borderMode=cv2.BORDER_REPLICATE)


def _tilt(frame, i, max_deg=8.0):
    """Out-of-plane tilt back and forth, like a phone rocking in a hand (perspective change)."""
    k = 0.5 * np.sin(np.radians(max_deg * np.sin(i / 10)))
    w, h = FRAME_W, FRAME_H
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, h * k / 2], [w, -h * k / 2], [w, h + h * k / 2], [0, h - h * k / 2]])
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (w, h),
                               borderMode=cv2.BORDER_REPLICATE)


MOTIONS = {"still": lambda f, i: f, "handheld": _handheld, "tilt": _tilt}


def synthetic_spoof_clips(image_dir, seconds, fps=30, seed=0):
    rng = np.random.default_rng(seed)
    paths = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in glob.glob(os.path.join(image_dir, ext)))
    for path in paths:
        img = cv2.imread(path)
        box = _face_box(img) if img is not None else None
        if box is None:
            print(f"  skip {os.path.basename(path)}: no face found")
            continue
        for face_px in (110, 170):
            base = _place_face(img, box, face_px)
            for sigma in (1.0, 2.0, 4.0):
                for motion, move in MOTIONS.items():
                    def frames(base=base, sigma=sigma, move=move):
                        for i in range(int(seconds * fps)):
                            f = move(base, i)
                            noise = rng.normal(0, sigma, f.shape)
                            yield np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                    yield {
                        "name": f"{os.path.splitext(os.path.basename(path))[0]}|{face_px}px|noise{sigma:g}|{motion}",
                        "label": "spoof", "kind": "synthetic", "fps": fps,
                        "face_px": face_px, "noise": sigma, "motion": motion, "frames": frames,
                    }


def recorded_clips(clip_dir, max_seconds):
    for label in ("live", "spoof"):
        for path in sorted(glob.glob(os.path.join(clip_dir, label, "*"))):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

            def frames(path=path, fps=fps):
                cap = cv2.VideoCapture(path)
                n = 0
                while n < max_seconds * fps:
                    ok, f = cap.read()
                    if not ok:
                        break
                    n += 1
                    yield cv2.resize(f, (FRAME_W, FRAME_H)) if f.shape[1] != FRAME_W else f
                cap.release()
            yield {"name": os.path.basename(path), "label": label, "kind": "recorded",
                   "fps": fps, "frames": frames}


# ------------------------------------------------------------------- evaluation
def evaluate_clip(clip):
    from app.models.tracked_person import TrackedPerson
    from app.processors.mediapipe_processor import MediaPipeProcessor
    from app.processors.person_analyzer import PersonAnalyzer

    _Clock.t = _dt.datetime(2026, 1, 1, 9, 0, 0)
    start = _Clock.t
    logger = _QuietLogger()
    mpp = MediaPipeProcessor()
    analyzer = PersonAnalyzer(logger)
    person, image_map, next_id = None, {}, 1
    n = flagged = with_face = 0
    first_flag = first_alert = None
    dt = _dt.timedelta(seconds=1.0 / clip["fps"])

    for frame in clip["frames"]():
        n += 1
        h, w = frame.shape[:2]
        results = mpp.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_res = results[0] if results else None
        if mp_res is not None:
            lm = mp_res["landmarks_object"].landmark
            xs = [p.x * w for p in lm]
            ys = [p.y * h for p in lm]
            box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            with_face += 1
            if person is None:
                person = TrackedPerson("Person-1", "Student", box, logger)
            person.update(box, "Student", mp_res)
        if person is not None:
            if mp_res is None:
                person.update(person.box, "Student", None)
            alerts, next_id = analyzer.analyze_liveness(person, frame, image_map, next_id)
            t = (_Clock.t - start).total_seconds()
            if person.is_spoofing:
                flagged += 1
                if first_flag is None:
                    first_flag = round(t, 2)
            if alerts and first_alert is None:
                first_alert = round(t, 2)
        _Clock.t += dt

    if hasattr(mpp, "close"):
        mpp.close()
    duration = n / clip["fps"]
    out = {k: v for k, v in clip.items() if k != "frames"}
    out.update({
        "seconds": round(duration, 1),
        "face_found_pct": round(100 * with_face / max(1, n), 1),
        "flagged_pct": round(100 * flagged / max(1, n), 1),
        "first_flag_s": first_flag,          # first frame classified as a photo
        "first_alert_s": first_alert,        # first "Image Detected" alert (evidence saved)
    })
    return out


def summarize(rows):
    def pct(a, b):
        return None if b == 0 else round(100.0 * a / b, 1)

    spoof = [r for r in rows if r["label"] == "spoof"]
    live = [r for r in rows if r["label"] == "live"]
    summary = {
        "spoof_clips": len(spoof),
        "spoof_detected_pct": pct(sum(r["first_alert_s"] is not None for r in spoof), len(spoof)),
        "spoof_median_time_to_alert_s": (float(np.median([r["first_alert_s"] for r in spoof if r["first_alert_s"] is not None]))
                                         if any(r["first_alert_s"] is not None for r in spoof) else None),
        "live_clips": len(live),
        "live_false_alarm_clip_pct": pct(sum(r["first_alert_s"] is not None for r in live), len(live)),
        "live_flagged_time_pct": (round(float(np.mean([r["flagged_pct"] for r in live])), 1) if live else None),
        "by_group": {},
    }
    for key in ("motion", "noise", "face_px"):
        for val in sorted({r.get(key) for r in spoof if r.get(key) is not None}, key=str):
            g = [r for r in spoof if r.get(key) == val]
            summary["by_group"][f"spoof {key}={val}"] = pct(sum(r["first_alert_s"] is not None for r in g), len(g))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="folder of face photos for synthetic spoof clips")
    ap.add_argument("--clips", help="folder with live/ and spoof/ recordings")
    ap.add_argument("--seconds", type=float, default=15.0, help="seconds evaluated per clip")
    ap.add_argument("--out", help="write full results JSON here")
    args = ap.parse_args()
    if not args.images and not args.clips:
        ap.error("give --images and/or --clips")

    _patch_clock()
    clips = []
    if args.images:
        clips += list(synthetic_spoof_clips(args.images, args.seconds))
    if args.clips:
        clips += list(recorded_clips(args.clips, args.seconds))

    rows, t0 = [], time.time()
    for i, clip in enumerate(clips, 1):
        r = evaluate_clip(clip)
        rows.append(r)
        alert = f"alert at {r['first_alert_s']}s" if r["first_alert_s"] is not None else "no alert"
        print(f"[{i:>3}/{len(clips)}] {r['label']:5} {r['name']:<40} face {r['face_found_pct']:5.1f}%  "
              f"flagged {r['flagged_pct']:5.1f}%  {alert}")

    summary = summarize(rows)
    print("\nSUMMARY")
    for k, v in summary.items():
        if k != "by_group":
            print(f"  {k}: {v}")
    for k, v in summary["by_group"].items():
        print(f"  {k}: {v}% detected")
    if summary["live_clips"] == 0:
        print("  (no live recordings given: false-alarm rate on real people NOT measured)")
    print(f"  wall time: {time.time() - t0:.0f}s")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "clips": rows}, f, indent=2)
        print(f"  results written to {args.out}")


if __name__ == "__main__":
    main()
