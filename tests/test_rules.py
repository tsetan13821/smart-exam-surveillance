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
