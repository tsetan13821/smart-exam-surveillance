# Smart Exam Surveillance: Technical Documentation

This document explains how the proctoring engine works, why the main design choices were made, and what has been measured. The [README](README.md) covers setup and day-to-day use.

Every number in the evaluation section comes from the benchmark scripts in `benchmarks/`, and the raw results are in `benchmarks/results/`. Anything that has **not** been measured is marked as such.

---

## 1. System overview

The system has three Streamlit apps and one real-time engine:

| Component | Entry point | Role |
|---|---|---|
| Student portal | `student_registration/student_portal.py` | Guided webcam enrollment: front, left and right poses, one ArcFace embedding each, saved as `pending` |
| Proctor dashboard | `review/dashboard.py` | Approves or rejects registrations; reviews sessions (incidents with evidence, attendance and breaks) |
| Session launcher | `launcher.py` | Selects approved students and sets the exam context (open-book, paper-based, allowed devices), then starts the engine |
| Proctoring engine | `main.py` → `app/core.py` | Real-time analysis of the webcam feed |

Data stays local:
- Enrollment embeddings are kept in `student_encodings.json`.
- Sessions and alerts are stored in SQLite (`proctoring_session.db`).
- Evidence screenshots are saved in `proctoring_logs/evidence/`.

---

## 2. Models

| Task | Model | Runtime |
|---|---|---|
| Face detection | SCRFD (`det_10g`, from InsightFace `buffalo_l`) | ONNX Runtime |
| Face identification | ArcFace ResNet-50 trained on WebFace600K (`w600k_r50`, from `buffalo_l`), 512-d embeddings | ONNX Runtime |
| Objects (person, phone, laptop, book, keyboard, mouse) | RT-DETR `PekingU/rtdetr_r50vd` (COCO classes) | PyTorch via Hugging Face Transformers |
| Face landmarks | MediaPipe FaceMesh with iris refinement (478 landmarks, up to 4 faces) | TFLite (MediaPipe 0.10.21) |

Only the detection and recognition models from `buffalo_l` are loaded. The pack's gender/age and landmark models were unused, yet they ran on every face.

**Identification.** Each face embedding is L2-normalised and compared by dot product with every enrolled embedding of the session's students (three per student). The best match is accepted at a cosine similarity of 0.50 or higher; anything below is `Unknown`.

**Head pose.** OpenCV `solvePnP` fits six landmarks (nose tip, chin, eye corners, mouth corners) to a canonical 3D face. Pitch and yaw come from the rotation, and the camera matrix is approximated from the frame size.

**Eye and mouth state.** The eye aspect ratio (EAR) and mouth aspect ratio (MAR) are computed from the landmarks. Landmarks keep sub-pixel precision, and the only preprocessing is CLAHE contrast equalisation; see section 4 for why.

---

## 3. Engine architecture

```
 capture thread                 inference thread                     processing thread
 ──────────────                 ────────────────                     ─────────────────
 webcam → 2-slot queue  ──────▶ newest frame ─▶ RT-DETR              every frame:
 (drops stale frames)                          InsightFace           ├─ MediaPipe FaceMesh (pose, gaze, EAR/MAR)
 displays latest frame          ─▶ latest results ─────────────────▶ ├─ tracker: link face boxes ↔ landmarks
                                                                     ├─ per-person: liveness, gaze, devices, pose
                                                                     ├─ session: presence, unknown faces, priority
                                                                     └─ draw overlay, then save evidence
```

- **Heavy models run asynchronously** (`HEAVY_MODELS_ASYNC`). RT-DETR and InsightFace take about a second per frame on a CPU. If they run in-line, every frame waits for them and blinks are missed (section 7.2). In async mode they work on the newest frame in their own thread, and everything else runs on every camera frame using their latest results.
- **CPU thread budget.** On CPU, PyTorch and ONNX Runtime each default to one busy-waiting thread per core. Running beside MediaPipe, that made every stage several times slower. In async mode on CPU, each heavy model is therefore limited to a quarter of the logical cores (`CPU_HEAVY_MODEL_THREADS`), and ONNX Runtime spin-waiting is disabled.
- **Tracking.** Face boxes are linked to FaceMesh results when the landmark nose tip falls inside the box (with a 20% margin). Persistent IDs come from centroid matching: 150 px for people, and 30 px for faces already flagged as photos so a photo can't take over a student's track. Tracks are dropped after 5 s unseen. A track that isn't matched in a frame has its landmarks cleared, so nothing is judged on a frozen face.
- **Per-face FaceMesh state.** Pupil smoothing and the landmark-presence buffer are kept separately for each face, matched by nose position across frames. Without that, several students' eye positions would be blended together.
- **Evidence after annotation.** Screenshots requested during analysis are written after the overlay is drawn, so the evidence shows the boxes and status that caused the alert.
- **Logging.** SQLite writes and debug logs go through a background logger thread.

---

## 4. Liveness (photo and screen spoofing)

### Design
A student has to show they are live by **blinking**:

1. A rolling 3 s median of the face's open-eye EAR is its baseline.
2. A blink starts when EAR falls below a fraction of that baseline, and completes when EAR recovers above 85% of it within 2 s.
3. That fraction is 70% for most faces. It drops to as low as 50% when the face's own landmark noise is high: the dip must be 4 robust standard deviations (MAD) below the baseline. If EAR stays low for longer than 2 s (looking down at a paper, for example), that becomes the new baseline, so blinks from that position still count.
4. A new face that hasn't blinked within **8 s**, or a confirmed face that hasn't blinked for **30 s**, is classified as a photo. "Image Detected" is raised once the classification has held for 2 s. Time without usable landmarks pauses the clock, so a face the system can't see is not judged.

Using each face's own baseline, rather than a fixed EAR threshold, keeps the check working for narrow eyes, glasses, and looking down.

### Why not head movement, optical flow or SSIM
The original detector accepted head-pose changes, gaze-label changes and optical-flow motion as proof of life, and used SSIM to catch static images. Measurements on static photo clips showed these signals sit inside the noise:

- Head pose jittered by 2–12° per frame, against a 2.5° "sway" threshold.
- The left/centre/right gaze label flipped about 16 times per second.
- Ordinary camera noise kept SSIM below its 0.98 "static" threshold, and kept optical flow above its 0.4 "motion" threshold.
- Head sway was measured against (0, 0) instead of the first reading, so the very first frame counted as a large movement.

A photo on a slightly moving phone also produces real pose changes and real pixel motion. None of these signals can tell a photo from a person, so the new detector doesn't use them.

Two preprocessing choices in the original pipeline amplified the landmark noise:
- Landmarks were rounded to integer pixels.
- A sharpening kernel was applied before FaceMesh.

Removing both, on static faces:

| Variant | Yaw jitter (sd) | EAR noise (relative sd) |
|---|---|---|
| Integer landmarks + sharpening (original) | 1.08° | 8.6% |
| Float landmarks, no sharpening | 0.29° | 3.1% |
| Float landmarks, CLAHE only (current) | 0.36° | 3.5% |

### Known gaps
- A **video replay** of the student blinking passes a blink check. Catching replays needs texture- or depth-based anti-spoofing (for example a MiniFASNet-style model), which isn't implemented.
- The **false-alarm rate on real people hasn't been measured** yet (section 7.1).

---

## 5. Rules and alerts

The proctor sets the exam context at launch:
- **Open-book** allows books.
- **Paper-based** treats looking down as normal; only looking sideways or up counts as looking away.
- **Allowed devices** are chosen from phone, laptop, mouse and keyboard.

Items are assigned to students in one of two ways:
- by desk polygon, if `calibrate_rois.py` has been run (the engine loads `desk_rois.json` for the session's students)
- otherwise, to the nearest student within 250 px

| Alert | Trigger (defaults in `config.py`) |
|---|---|
| Image Detected | No blink within 8 s of appearing, or for 30 s after the last blink; held for 2 s. A face flagged as a photo doesn't count as the student being present. |
| Suspicious Activity (Device + Gaze) | Non-allowed device assigned to the student, and looking away (yaw > 25° or pitch beyond 20°) for 3 s |
| Contraband Detected | Non-allowed device seen at the student's desk for at least 2 s (gaps of up to 3 s tolerated) |
| Sustained Gaze Away | Looking away for 5 s with no device involved |
| Unauthorized Face | Face matching no enrolled student, for 3 s |
| Student Missing | Present student unseen for 10 s, or no-show 60 s into the session |
| Peeking Alert | Head turned more than 25° sideways for 5 s |
| Excessive Movement | More than 20 large head movements within 60 s (each one a shift of more than 10° in the median-smoothed head pose) |

The status bar shows one classification per frame, in this priority order: Image Detected, Student Missing, Unauthorized Face, Device + Gaze, Contraband, Peeking, Excessive Movement, Sustained Gaze Away, Normal. Each incident saves one evidence screenshot. The trigger re-arms when the condition clears.

Attendance events (marked present, left, returned) are logged as evidence and used for the break report in the dashboard.

---

## 6. Changes from the first version

End-to-end testing of the original code found several bugs, and each one was fixed:

| Area | Problem | Fix |
|---|---|---|
| Liveness | Photos were accepted as live (section 4) | Adaptive blink liveness; sub-pixel landmarks without sharpening |
| Throughput | Heavy models ran in-line, so 2.2 frames/s were analysed with up to 2.5 s gaps | Background inference thread with a CPU thread budget |
| Multi-student | FaceMesh tracked one face, so other students got no pose, gaze or blink analysis | Up to 4 faces, with per-face temporal state |
| Gaze rule | Looking away without a device never raised an alert | "Sustained Gaze Away" after 5 s |
| Paper exams | Looking down at the paper counted as looking away | Paper-based exam option |
| Contraband | A single false detection created evidence, and evidence was saved on every frame | 2 s persistence; one screenshot per incident |
| Photo IDs | "Image-N" counter incremented on every frame | Increments once per photo |
| Absence | A student who left was judged on frozen landmarks and reported missing late | Stale landmarks cleared; missing 10 s after leaving; no-show after 60 s |
| Setup | Unapproved students could be put in a session, and saved desk regions were never loaded | Engine rejects unapproved students and loads `desk_rois.json` |
| Device classes | Rules referenced earphones and tablets, which COCO can't detect | Limited to classes the detector supports |
| Evidence | Screenshots were taken before annotation | Saved after drawing |

---

## 7. Evaluation

All measurements were taken on one laptop CPU: an Intel Core 13th-gen mobile processor with 16 logical cores and no GPU, running Windows 11 and Python 3.11.

### 7.1 Liveness on photo attacks

**Method.** `benchmarks/liveness_benchmark.py` runs the real MediaPipe processor and liveness code on a simulated 30 FPS clock. The test set is 126 synthetic spoof clips of 15 s each. They are built from 7 face photographs, crossed with:
- face width: 110 or 170 px in a 640×480 frame
- Gaussian camera noise: σ = 1, 2 or 4
- motion: held still, hand-held shake (about 3 px and 1.5°), or tilting back and forth up to 8° out of plane

A clip counts as detected if "Image Detected" is raised within its 15 s.

| | Original version | Current version |
|---|---|---|
| Photo clips detected | **0 / 126 (0%)** | **126 / 126 (100%)** |
| Median time to alert | — | 10.07 s (8 s blink window + 2 s hold) |
| Still / hand-held / tilting | 0% / 0% / 0% | 100% / 100% / 100% |
| Noise σ = 1 / 2 / 4 | 0% / 0% / 0% | 100% / 100% / 100% |
| Face 110 px / 170 px | 0% / 0% | 100% / 100% |

The original version confirmed every photo as live on its first frame, through the head-sway bug. Optical flow from sensor noise then kept resetting its timer. An earlier run of the current design, before the noise-adaptive threshold was added, detected 119 of 126 (94.4%). All 7 misses were the same narrow-eyed photo, whose EAR jitters by about 14% on a small face, so single-frame dips passed for blinks. Scaling the required dip to each face's own noise fixed all 7.

**Not yet measured:** how often real, blinking students are wrongly flagged. That needs recordings of real people, and none are in the repository: `benchmarks/record_clip.py` records them, and the benchmark reads them with `--clips`. Synthetic clips also leave out printed-photo glare, screen moiré and real hand tremor.

### 7.2 Throughput

**Method.** `benchmarks/perf_benchmark.py` feeds the full pipeline (all models, tracker, rules and drawing) with frames paced like a 30 FPS webcam for 60 s, using a single-student frame. It measures:
- how many frames per second actually pass through MediaPipe, liveness and the rules
- the longest gap between two analysed frames, since a blink shorter than this gap can be missed entirely
- how often the RT-DETR and InsightFace results are refreshed

| | Original version (in-line, every 5th frame) | Current version (async, thread budget) |
|---|---|---|
| Frames analysed per second | 2.2 | **30.0** (every camera frame) |
| Longest gap between analysed frames | 2,511 ms | **52 ms** |
| Gap, 95th percentile | 2,079 ms | 40 ms |
| Detection / identity refreshes per second | 0.44 | 0.87 |
| RT-DETR per inference | 1,055 ms | 946 ms |
| InsightFace per inference | 789 ms (all 5 models) | 198 ms (detection + recognition only) |
| MediaPipe per frame | 44.5 ms | 12.7 ms |
| Liveness + rules per frame | 34.9 ms (optical flow, SSIM) | 0.3 ms |

A typical blink lasts 100–400 ms. With a 52 ms maximum gap, every blink spans at least one analysed frame, which wasn't true with gaps of over 2 s.

Without the thread budget, async mode on this CPU analysed only 10.5 frames/s and refreshed detections once every ~12 s. PyTorch, ONNX Runtime and MediaPipe were all competing for every core.

Not measured: GPU throughput (no GPU was available).

### 7.3 Automated tests

`pytest tests` runs 35 tests in about 15 s without loading any model. A simulated clock drives the rule engine, liveness detector, tracker and logger, so time-based behaviour is checked to the frame:
- each alert fires at its threshold and not before
- open-book and paper-exam contexts behave correctly
- desk regions assign items to the right student
- evidence is saved once per incident
- photo noise never counts as a blink, while real blinks count even at 6 FPS or on noisy faces
- missing and no-show timing is correct

`main.py` was also run end to end with a fake 30 FPS camera showing a photo of an enrolled student. In order, it:
- marked attendance
- flagged the photo
- stopped counting the photo as the student, then reported the student missing
- shut down every thread cleanly
- saved annotated evidence for each alert

---

## 8. Limitations

- **False alarms on real students are unmeasured** (section 7.1). Long stretches without blinking (more than 30 s) would be flagged.
- **Video replays** of a blinking student defeat blink-based liveness.
- **Device coverage** is limited to COCO classes. Earphones, calculators, smartwatches and tablets need a custom-trained detector.
- **Gaze** is estimated from head pose, plus a coarse left/centre/right label from the nose position relative to the eyes. Eye-only glances with the head still are not tracked.
- **One camera.** Each session analyses one webcam feed. FaceMesh is configured for up to 4 faces.
- **Identity matching** uses a single cosine-similarity threshold (0.50), not calibrated on a held-out set of enrolled students.

---

## 9. Reproducing the numbers

```bash
pytest tests

# Liveness (synthetic photo clips from a folder of face photos; add --clips for recordings)
python benchmarks/liveness_benchmark.py --images path/to/face_photos --seconds 15 --out liveness.json

# Throughput (paced like a 30 FPS webcam)
python benchmarks/perf_benchmark.py --source path/to/student_frame.jpg --seconds 60 --out perf.json
```

The results quoted above are stored in `benchmarks/results/`:
- `liveness_before.json` and `liveness_after.json`
- `perf_before.json` and `perf_after.json`
