"""Liveness detector: adaptive blink detection, timing, noise robustness."""
import numpy as np

from sim import blinking_ear, mp_result, run_session

FPS = 30


def make_person(logger):
    from app.models.tracked_person import TrackedPerson
    return TrackedPerson("Person-1", "Test Student", [250, 150, 360, 300], logger)


def drive(clock, person, seconds, mp_fn, fps=FPS):
    """Feed check_liveness for `seconds`; return [(t, is_spoofing)] at each change."""
    changes, last = [], None
    for i in range(int(seconds * fps)):
        t = i / fps
        spoof, _ = person.liveness_detector.check_liveness(mp_fn(t))
        if spoof != last:
            changes.append((round(t, 2), spoof))
            last = spoof
        clock.advance(1 / fps)
    return changes


def noisy_ear(rng, level=0.30, rel_sd=0.05):
    """Photo-like EAR: constant plus landmark noise (5% sd, worse than the 3.5% measured)."""
    return level * (1 + rng.normal(0, rel_sd))


def test_photo_is_flagged_after_confirm_window(clock, logger):
    rng = np.random.default_rng(0)
    p = make_person(logger)
    changes = drive(clock, p, 20, lambda t: mp_result(ear=noisy_ear(rng)))
    flagged_at = next(t for t, s in changes if s)
    assert 7.9 <= flagged_at <= 8.3
    assert p.liveness_state == "UNCONFIRMED"


def test_photo_noise_never_counts_as_a_blink(clock, logger):
    rng = np.random.default_rng(1)
    p = make_person(logger)
    changes = drive(clock, p, 120, lambda t: mp_result(ear=noisy_ear(rng), yaw=rng.normal(0, 1.0)))
    assert changes[-1][1] is True                     # still flagged at the end
    assert sum(1 for _, s in changes if not s) == 1   # never went back to live


def test_narrow_eyed_noisy_photo_is_still_flagged(clock, logger):
    """Small, narrow-eyed face (EAR ~0.05): jitter is ~14% of EAR, beyond the fixed 30% drop's margin."""
    rng = np.random.default_rng(5)
    p = make_person(logger)
    changes = drive(clock, p, 30, lambda t: mp_result(ear=noisy_ear(rng, level=0.05, rel_sd=0.14)))
    assert changes[-1][1] is True
    assert sum(1 for _, s in changes if not s) == 1


def test_noisy_face_still_confirms_with_real_blinks(clock, logger):
    rng = np.random.default_rng(6)
    p = make_person(logger)
    changes = drive(clock, p, 40, lambda t: mp_result(ear=blinking_ear(t, period=4.0) * (1 + rng.normal(0, 0.12))))
    assert not any(s for _, s in changes)


def test_blinking_person_is_never_flagged(clock, logger):
    rng = np.random.default_rng(2)
    p = make_person(logger)
    changes = drive(clock, p, 60, lambda t: mp_result(ear=blinking_ear(t, period=4.0) * (1 + rng.normal(0, 0.03))))
    assert not any(s for _, s in changes)
    assert p.liveness_state == "CONFIRMED_LIVE"


def test_concentrating_student_has_30s_grace(clock, logger):
    """Blinks normally for 10 s, then stops blinking: flagged only ~30 s after the last blink."""
    p = make_person(logger)
    last_blink = 9.0
    changes = drive(clock, p, 45, lambda t: mp_result(ear=blinking_ear(t, period=3.0) if t < 10 else 0.30))
    flagged_at = next(t for t, s in changes if s)
    assert 29.5 <= flagged_at - last_blink <= 31.5


def test_blink_after_being_flagged_restores_live(clock, logger):
    p = make_person(logger)
    changes = drive(clock, p, 15, lambda t: mp_result(ear=0.30 if t < 12 else blinking_ear(t, period=1.0)))
    assert changes[-1] == (changes[-1][0], False) and any(s for _, s in changes)


def test_blinks_detected_while_looking_down(clock, logger):
    """Looking down lowers EAR for a long time; blinks from that lower level must still count."""
    def ear(t):
        if t < 3:
            return blinking_ear(t, period=2.0)
        return blinking_ear(t, period=5.0, open_ear=0.17, closed_ear=0.06)
    p = make_person(logger)
    changes = drive(clock, p, 60, lambda t: mp_result(ear=ear(t), pitch=25.0))
    assert not any(s for _, s in changes)


def test_face_without_landmarks_is_not_judged(clock, logger):
    p = make_person(logger)
    changes = drive(clock, p, 30, lambda t: None)
    assert not any(s for _, s in changes)
    assert p.liveness_state == "UNCONFIRMED"


def test_low_frame_rate_still_catches_blinks(clock, logger):
    """At ~6 processed FPS a 200 ms blink spans one or two frames; it must still register."""
    p = make_person(logger)
    changes = drive(clock, p, 40, lambda t: mp_result(ear=blinking_ear(t, period=4.0, duration=0.2)), fps=6)
    assert not any(s for _, s in changes)


def test_large_head_movement_uses_first_reading_as_reference(clock, logger):
    p = make_person(logger)
    det = p.liveness_detector
    rng = np.random.default_rng(3)
    events = [det.check_liveness(mp_result(yaw=-15 + rng.normal(0, 1.0), ear=0.3))[1] for _ in range(60)]
    assert not any(events)                            # sitting turned 15 deg is not "movement"
    events = [det.check_liveness(mp_result(yaw=5 + rng.normal(0, 1.0), ear=0.3))[1] for _ in range(30)]
    assert sum(events) == 1                           # one 20 deg turn = one movement


def test_photo_raises_image_detected_alert_with_evidence(clock, logger):
    rng = np.random.default_rng(4)
    run = run_session(clock, logger, 15, lambda t: (mp_result(ear=noisy_ear(rng)), []))
    t = run.first("Image Detected")
    assert t is not None and 7.9 <= t <= 8.3
    ev = logger.first_evidence("IMAGE_DETECTED")
    assert ev is not None and 9.9 <= ev <= 10.4
