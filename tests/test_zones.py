"""雙站位區：左右各一區，各自一個鎖定槽，互不影響。"""

from __future__ import annotations

import pytest

from booth_tracker.lock import LockConfig, PlayerLock, side_by_side
from tests.scene import person, player_x, run

Z = 2.5
ZONES = side_by_side(center_x=0.0, z=Z, radius=0.5, gap=0.4)  # 中心在 x = -0.7、+0.7
L, R = -0.7, 0.7


def new_lock() -> PlayerLock:
    return PlayerLock(LockConfig(zones=ZONES))


def both_claimed() -> tuple[PlayerLock, float]:
    lock = new_lock()
    frames = run(lock, lambda t: [person(L, Z, hands_up=True), person(R, Z, hands_up=True)], 1.2)
    assert [f.mode for f in frames[-1][1]] == ["track", "track"]
    return lock, frames[-1][0] + 1 / 30


def test_side_by_side_leaves_gap_between_zones():
    left, right = ZONES
    assert right.x - left.x == pytest.approx(left.r_in + right.r_in + 0.4)
    assert left.r_out <= left.r_in + 0.2  # 出區遲滯不越過空白帶中線


def test_frames_are_ordered_by_zone_id():
    lock = new_lock()
    frames = lock.update([], 0.0)
    assert [f.zone_id for f in frames] == [0, 1]


def test_each_zone_claims_independently():
    lock = new_lock()
    last = run(lock, lambda t: [person(L, Z, hands_up=True), person(R, Z)], 1.2)[-1][1]
    assert last[0].mode == "track"
    assert last[1].mode == "idle"
    assert last[1].occupied


def test_both_zones_get_distinct_lock_ids():
    lock, _ = both_claimed()
    frames = lock.update([person(L, Z), person(R, Z)], 1.3)
    assert frames[0].lock_id != frames[1].lock_id
    assert abs(player_x(frames[0]) - L) < 0.05
    assert abs(player_x(frames[1]) - R) < 0.05


def test_hold_in_one_zone_does_not_affect_other():
    lock, t0 = both_claimed()
    frames = run(lock, lambda t: [person(R, Z)], 1.0, t0)
    for _, (left, right), _ in frames:
        assert left.mode == "hold"
        assert right.mode == "track"


def test_player_crossing_into_other_zone_keeps_original_lock():
    lock = new_lock()
    run(lock, lambda t: [person(L, Z, hands_up=True)], 1.2)
    t0 = 1.2
    # 左區玩家走進右區中央，還舉手
    walk = lambda t: [person(min(R, L + (t - t0) * 2.0), Z, hands_up=t > t0 + 0.8)]
    frames = run(lock, walk, 2.5, t0)
    left, right = frames[-1][1]
    assert left.mode == "track" and left.lock_id == 1
    assert not left.in_zone
    assert right.mode == "idle" and right.lock_id is None
    assert not right.occupied


def test_two_claimants_in_one_zone_do_not_block_other_zone():
    lock = new_lock()
    scene = lambda t: [person(L - 0.2, Z, hands_up=True), person(L + 0.2, Z, hands_up=True), person(R, Z, hands_up=True)]
    left, right = run(lock, scene, 1.2)[-1][1]
    assert left.mode == "idle"
    assert right.mode == "track"


def test_release_one_zone_only():
    lock, t0 = both_claimed()
    lock.release(zone_id=0)
    left, right = lock.update([person(L, Z), person(R, Z)], t0)
    assert left.mode == "idle"
    assert right.mode == "track"
