"""Rule-engine behaviour: device, gaze, exam context, peeking, absence, desk ROIs."""
from sim import blinking_ear, item, mp_result, run_session

LOOKING_DOWN = 30.0   # pitch beyond CHEATING_GAZE_AWAY_THRESHOLD_PITCH (20)


def live(t, **kw):
    """A live, blinking student."""
    return mp_result(ear=blinking_ear(t), **kw)


def test_phone_plus_looking_down_escalates_after_3s(clock, logger):
    run = run_session(clock, logger, 8, lambda t: (live(t, pitch=LOOKING_DOWN), [item("cell phone")]))
    t = run.first("Suspicious Activity (Device + Gaze)")
    assert t is not None and 2.9 <= t <= 3.3
    assert "SUSPICIOUS_DEVICE_GAZE" in logger.evidence_types()


def test_phone_on_desk_is_contraband(clock, logger):
    run = run_session(clock, logger, 6, lambda t: (live(t), [item("cell phone")]))
    t = run.first("Contraband Detected")
    assert t is not None and t <= 3.0
    assert "CONTRABAND_DETECTED" in logger.evidence_types()


def test_open_book_exam_allows_books(clock, logger):
    ctx = {"is_open_book": True, "allowed_devices": ["book"]}
    run = run_session(clock, logger, 6, lambda t: (live(t), [item("book")]), exam_context=ctx)
    assert run.statuses() == {"Normal"}


def test_open_book_exam_still_flags_phones(clock, logger):
    ctx = {"is_open_book": True, "allowed_devices": ["book"]}
    run = run_session(clock, logger, 6, lambda t: (live(t), [item("cell phone")]), exam_context=ctx)
    assert "Contraband Detected" in run.statuses()


def test_explicitly_allowed_device_is_not_flagged(clock, logger):
    ctx = {"is_open_book": False, "allowed_devices": ["keyboard"]}
    run = run_session(clock, logger, 6, lambda t: (live(t), [item("keyboard")]), exam_context=ctx)
    assert run.statuses() == {"Normal"}


def test_peeking_alert_after_5s(clock, logger):
    run = run_session(clock, logger, 8, lambda t: (live(t, yaw=35.0), []))
    t = run.first("Peeking Alert")
    assert t is not None and 4.9 <= t <= 5.3
    assert "HEAD_TURN_ALERT" in logger.evidence_types()


def test_normal_student_raises_nothing(clock, logger):
    run = run_session(clock, logger, 20, lambda t: (live(t), []))
    assert run.statuses() == {"Normal"}
    assert logger.evidence_types() == ["ATTENDANCE_MARKED"]


def test_student_leaving_raises_missing_after_10s(clock, logger):
    run = run_session(clock, logger, 17, lambda t: (live(t), []), present_fn=lambda t: t < 2.0)
    t = run.first("Student Missing")
    assert t is not None and 11.9 <= t <= 12.3          # left at 2.0 s
    assert {"PRESENCE_LEFT", "STUDENT_MISSING"} <= set(logger.evidence_types())


def test_desk_roi_ignores_items_on_other_desks(clock, logger):
    rois = {"Test Student": [[200, 300], [420, 300], [420, 470], [200, 470]]}
    far_phone = item("cell phone", box=[560, 380, 620, 440])
    run = run_session(clock, logger, 6, lambda t: (live(t), [far_phone]), rois=rois)
    assert run.statuses() == {"Normal"}


def test_desk_roi_flags_items_on_own_desk(clock, logger):
    rois = {"Test Student": [[200, 300], [420, 300], [420, 470], [200, 470]]}
    run = run_session(clock, logger, 6, lambda t: (live(t), [item("cell phone")]), rois=rois)
    assert "Contraband Detected" in run.statuses()


# --- Fixes ---------------------------------------------------------------

def test_looking_away_alone_raises_alert_after_5s(clock, logger):
    run = run_session(clock, logger, 10, lambda t: (live(t, pitch=LOOKING_DOWN), []))
    t = run.first("Sustained Gaze Away")
    assert t is not None and 4.9 <= t <= 5.3
    assert logger.evidence_types().count("GAZE_AWAY") == 1


def test_paper_exam_allows_looking_down(clock, logger):
    ctx = {"is_paper_exam": True, "allowed_devices": []}
    run = run_session(clock, logger, 20, lambda t: (live(t, pitch=LOOKING_DOWN), []), exam_context=ctx)
    assert run.statuses() == {"Normal"}


def test_paper_exam_still_flags_looking_up(clock, logger):
    ctx = {"is_paper_exam": True, "allowed_devices": []}
    run = run_session(clock, logger, 10, lambda t: (live(t, pitch=-LOOKING_DOWN), []), exam_context=ctx)
    assert "Sustained Gaze Away" in run.statuses()


def test_single_frame_phone_detection_is_ignored(clock, logger):
    """A one-off detection (typically a false positive) must not raise contraband."""
    run = run_session(clock, logger, 8, lambda t: (live(t), [item("cell phone")] if 1.0 <= t < 1.2 else []))
    assert run.statuses() == {"Normal"}
    assert "CONTRABAND_DETECTED" not in logger.evidence_types()


def test_device_must_persist_2s_before_contraband(clock, logger):
    run = run_session(clock, logger, 6, lambda t: (live(t), [item("cell phone")]))
    t = run.first("Contraband Detected")
    assert t is not None and 1.9 <= t <= 2.2


def test_phone_held_low_is_flagged_when_student_is_alone(clock, logger):
    """A phone in the lap is far below the face (here ~280 px); with one person in view it is still theirs."""
    lap_phone = item("cell phone", box=[420, 440, 480, 480])
    run = run_session(clock, logger, 6, lambda t: (live(t), [lap_phone]))
    t = run.first("Contraband Detected")
    assert t is not None and 1.9 <= t <= 2.2


def _person(pid, box, logger):
    from app.models.tracked_person import TrackedPerson
    return TrackedPerson(pid, pid, box, logger)


def test_device_goes_to_the_nearest_of_several_people(logger):
    from app.processors.proctoring_analyzer import ProctoringAnalyzer
    left = _person("Left", [100, 100, 200, 220], logger)      # faces 120 px tall
    right = _person("Right", [800, 100, 900, 220], logger)
    people = {"Left": left, "Right": right}
    assert ProctoringAnalyzer.device_owner((870, 600), people) is right   # ~440 px below, within 4 face heights
    assert ProctoringAnalyzer.device_owner((150, 500), people) is left


def test_device_far_from_everyone_belongs_to_nobody(logger):
    from app.processors.proctoring_analyzer import ProctoringAnalyzer
    people = {"A": _person("A", [100, 100, 160, 160], logger), "B": _person("B", [300, 100, 360, 160], logger)}
    assert ProctoringAnalyzer.device_owner((1200, 700), people) is None      # beyond 4 face heights and 250 px


def test_evidence_logged_once_per_incident(clock, logger):
    """Phone on the desk for 6 s, away for 3 s, back for 6 s = two incidents, two screenshots."""
    on = lambda t: t < 6 or t >= 9
    run_session(clock, logger, 15, lambda t: (live(t), [item("cell phone")] if on(t) else []))
    assert logger.evidence_types().count("CONTRABAND_DETECTED") == 2


def test_image_ids_increment_once_per_photo(clock, logger):
    run = run_session(clock, logger, 25, lambda t: (mp_result(ear=0.30), []))
    assert "Image Detected" in run.statuses()
    assert run.analyzer.next_image_id == 2


def test_missing_alert_10s_after_leaving_even_while_tracker_holds_track(clock, logger):
    run = run_session(clock, logger, 17, lambda t: (live(t), []),
                      present_fn=lambda t: t < 2.0, keep_stale=True)
    t = run.first("Student Missing")
    assert t is not None and 13.9 <= t <= 14.3      # 2 s grace + 10 s threshold after leaving at 2 s


def test_student_who_never_arrives_is_flagged(clock, logger):
    run = run_session(clock, logger, 70, lambda t: (live(t), []), present_fn=lambda t: False)
    t = run.first("Student Missing")
    assert t is not None and 59.9 <= t <= 60.3
    assert any("has not appeared" in m for _, a, m in logger.evidence if a == "STUDENT_MISSING")
