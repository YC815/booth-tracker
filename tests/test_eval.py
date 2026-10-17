from __future__ import annotations

import numpy as np

from booth_tracker.evaluate import annotate, evaluate
from booth_tracker.tracker import BoothTracker
from tests.synth import detection
from tests.test_tracker import VENUE, FakeDetector, FakeSource


def claim_then_vanish(t: float):
    if t < 1.3:
        return [detection(0.0, 2.5, hands_up=True)]
    if t < 2.0:
        return [detection(0.0, 2.5)]
    return []


def test_evaluate_reports_claim_and_hold_events():
    tracker = BoothTracker(FakeSource(None, 3.0), VENUE, detector=FakeDetector(claim_then_vanish))
    report = evaluate(tracker)
    kinds = [(e.zone_id, e.what) for e in report.events]
    assert (0, "認領 lock_id=1") in kinds
    assert (0, "track → hold") in kinds
    z = report.zones[0]
    assert z.claims == 1
    assert z.frames["track"] > 0 and z.frames["hold"] > 0
    assert report.frames == 90


def test_report_text_mentions_each_zone():
    tracker = BoothTracker(FakeSource(None, 1.5), VENUE, detector=FakeDetector(claim_then_vanish))
    text = evaluate(tracker).text()
    assert "zone 0" in text and "認領 1 次" in text


def test_annotate_draws_on_a_copy():
    tracker = BoothTracker(FakeSource(None, 1.3), VENUE, detector=FakeDetector(claim_then_vanish))
    frames = []
    evaluate(tracker, on_frame=lambda tf: frames.append(tf))
    tf = frames[-1]
    img = annotate(tf, VENUE)
    assert img.shape == tf.frame.rgb.shape
    assert img.dtype == np.uint8
    assert not np.array_equal(img[..., ::-1], tf.frame.rgb)  # 有畫東西
    assert (tf.frame.rgb == 200).all()  # 原圖沒被改
