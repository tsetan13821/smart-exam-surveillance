"""
Record a short webcam clip for the liveness benchmark.

  python benchmarks/record_clip.py live  still      # sit normally, as in an exam
  python benchmarks/record_clip.py live  writing    # look down and write
  python benchmarks/record_clip.py spoof phone_photo # hold a photo of a face up to the camera

Clips are saved to benchmarks/clips/<label>/<name>.mp4 (git-ignored: they contain faces).
A 3-second countdown runs before recording starts. Press q to stop early.
"""
import argparse
import os
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", choices=["live", "spoof"])
    ap.add_argument("name", help="short name for the clip, e.g. still, writing, phone_photo")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "benchmarks", "clips", args.label)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{args.name}.mp4")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    start = time.time()
    while time.time() - start < 3:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(frame, f"Recording in {3 - int(time.time() - start)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("record_clip", frame)
        cv2.waitKey(1)

    writer = None
    start, n = time.time(), 0
    while time.time() - start < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        n += 1
        preview = frame.copy()
        cv2.putText(preview, f"REC {time.time() - start:4.1f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("record_clip", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Saved {n} frames ({n / fps:.1f}s at {fps:.0f} fps) to {path}")


if __name__ == "__main__":
    main()
