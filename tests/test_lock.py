"""鎖定狀態機的情境測試。站位區中心 (x=0, z=2.5)，半徑進 0.7 m、出 1.0 m。"""

from __future__ import annotations

from booth_tracker.lock import LockConfig, PlayerLock
from tests.scene import person, player_x, run

TAPE_Z = 2.5


def new_lock() -> PlayerLock:
    return PlayerLock(LockConfig(zone_x=0.0, zone_z=TAPE_Z))


def claimed_lock() -> tuple[PlayerLock, float]:
    """玩家站膠帶舉手 1.2 s 完成認領，回傳 (lock, 下一幀時間)。"""
    lock = new_lock()
    frames = run(lock, lambda t: [person(0, TAPE_Z, hands_up=True)], 1.2)
    assert frames[-1][1].mode == "track"
    return lock, frames[-1][0] + 1 / 30


# ---- 認領 ----


def test_hands_up_in_zone_claims():
    lock = new_lock()
    frames = run(lock, lambda t: [person(0, TAPE_Z, hands_up=True)], 1.2)
    last = frames[-1][1]
    assert last.mode == "track"
    assert last.lock_id == 1
    assert last.player_index == 0


def test_standing_without_hands_up_never_claims():
    lock = new_lock()
    frames = run(lock, lambda t: [person(0, TAPE_Z)], 5)
    assert all(f.mode == "idle" and f.lock_id is None for _, f, _ in frames)


def test_hands_up_outside_zone_does_not_claim():
    lock = new_lock()
    frames = run(lock, lambda t: [person(1.5, TAPE_Z, hands_up=True)], 3)
    assert all(f.mode == "idle" for _, f, _ in frames)


def test_two_claimants_in_zone_do_not_claim():
    lock = new_lock()
    scene = lambda t: [person(-0.3, TAPE_Z, hands_up=True), person(0.3, TAPE_Z, hands_up=True)]
    frames = run(lock, scene, 3)
    assert all(f.mode == "idle" for _, f, _ in frames)


def test_walking_with_hands_up_does_not_claim():
    lock = new_lock()
    frames = run(lock, lambda t: [person(-0.6 + t * 1.0, TAPE_Z, hands_up=True)], 1.2)
    assert all(f.mode == "idle" for _, f, _ in frames)


def test_person_without_depth_cannot_claim():
    lock = new_lock()
    frames = run(lock, lambda t: [person(0, TAPE_Z, hands_up=True, has_depth=False)], 3)
    assert all(f.mode == "idle" for _, f, _ in frames)


def test_without_depth_requirement_estimated_person_can_claim():
    lock = PlayerLock(LockConfig(zone_x=0.0, zone_z=TAPE_Z, require_depth=False))
    frames = run(lock, lambda t: [person(0, TAPE_Z, hands_up=True, has_depth=False)], 1.2)
    assert frames[-1][1].mode == "track"


def test_claim_progress_reports_partial():
    lock = new_lock()
    frames = run(lock, lambda t: [person(0, TAPE_Z, hands_up=True)], 0.5)
    assert 0.3 < frames[-1][1].claim_progress < 0.7


def test_static_figure_in_zone_does_not_block_claim():
    lock = new_lock()
    scene = lambda t: [person(0.3, TAPE_Z), person(-0.3, TAPE_Z, hands_up=True)]
    last = run(lock, scene, 1.2)[-1][1]
    assert last.mode == "track"
    assert abs(player_x(last) + 0.3) < 0.05


# ---- 鎖定後路人 ----


def test_passerby_behind_does_not_steal():
    lock, t0 = claimed_lock()
    scene = lambda t: [person(0, TAPE_Z), person(-2 + (t - t0) * 1.5, TAPE_Z + 1.0)]
    frames = run(lock, scene, 3, t0)
    for _, f, _ in frames:
        assert f.mode == "track" and f.lock_id == 1
        assert abs(player_x(f)) < 0.05


def test_passerby_crossing_close_never_becomes_player():
    lock, t0 = claimed_lock()
    scene = lambda t: [person(0, TAPE_Z), person(-2 + (t - t0) * 1.5, TAPE_Z + 0.3)]
    frames = run(lock, scene, 3, t0)
    for _, f, _ in frames:
        assert f.lock_id == 1
        assert abs(player_x(f)) < 0.05
    assert frames[-1][1].mode == "track"


def test_player_moving_fast_inside_zone_keeps_tracking():
    lock, t0 = claimed_lock()
    # 閃身速度 3 m/s 往右 0.2 s（0 -> 0.6 m，還在站位區內）
    frames = run(lock, lambda t: [person((t - t0) * 3.0, TAPE_Z)], 0.2, t0)
    assert all(f.mode == "track" and f.lock_id == 1 for _, f, _ in frames)


def test_occluder_in_front_holds_then_player_returns():
    lock, t0 = claimed_lock()
    # 遮擋者從左邊走到玩家正前方 0.7 m（玩家還看得到）
    walk_in = lambda t: [person(0, TAPE_Z), person(-2 + (t - t0) * 2.0, TAPE_Z - 0.7)]
    run(lock, walk_in, 1.0, t0)
    # 擋住 1.5 s：只看得到遮擋者
    t1 = t0 + 1.0
    blocked = run(lock, lambda t: [person(0, TAPE_Z - 0.7)], 1.5, t1)
    for _, f, _ in blocked:
        assert f.lock_id == 1
        assert f.mode == "hold"
        assert abs(player_x(f)) < 0.05
    # 遮擋者離開、玩家重新出現
    t2 = t1 + 1.5
    back = run(lock, lambda t: [person(0, TAPE_Z)], 1.0, t2)
    assert back[-1][1].mode == "track"
    assert back[-1][1].lock_id == 1


def test_occluder_flicker_in_front_cannot_rejoin_as_player():
    """遮擋者在玩家消失後才被偵測到（新軌跡），但比玩家近 0.5 m，不能被認成玩家。"""
    lock, t0 = claimed_lock()
    frames = run(lock, lambda t: [person(0, TAPE_Z - 0.5)], 2.5, t0)
    for _, f, persons in frames:
        assert f.lock_id == 1
        assert f.mode in ("hold", "paused")
        assert abs(player_x(f)) < 0.05


def test_long_occlusion_pauses_then_idles():
    lock, t0 = claimed_lock()
    cfg = lock.config
    frames = run(lock, lambda t: [person(0, TAPE_Z - 0.7)], cfg.hold_s + cfg.pause_s + 0.5, t0)
    modes = [f.mode for _, f, _ in frames]
    assert "paused" in modes
    assert modes[-1] == "idle"
    assert frames[-1][1].lock_id is None


# ---- 玩家消失 / 離開 ----


def test_short_dropout_keeps_tracking():
    lock, t0 = claimed_lock()
    scene = lambda t: [] if t0 + 0.5 <= t < t0 + 0.7 else [person(0, TAPE_Z)]
    frames = run(lock, scene, 1.0, t0)
    assert frames[-1][1].mode == "track"
    assert all(f.lock_id == 1 for _, f, _ in frames)


def test_dropout_in_place_rejoins():
    lock, t0 = claimed_lock()
    scene = lambda t: [] if t0 + 0.5 <= t < t0 + 1.5 else [person(0, TAPE_Z)]
    frames = run(lock, scene, 2.5, t0)
    assert frames[-1][1].mode == "track"
    assert all(f.lock_id == 1 for _, f, _ in frames)


def test_player_out_of_zone_holds_and_returns():
    lock, t0 = claimed_lock()
    out_and_back = lambda t: [person(min(1.4, (t - t0) * 2.0), TAPE_Z)] if t < t0 + 1.5 else [
        person(max(0.0, 1.4 - (t - t0 - 1.5) * 2.0), TAPE_Z)
    ]
    frames = run(lock, out_and_back, 2.5, t0)
    assert any(f.mode == "hold" for _, f, _ in frames)
    assert frames[-1][1].mode == "track"
    assert all(f.lock_id == 1 for _, f, _ in frames)


def test_newcomer_on_tape_cannot_inherit_after_player_walks_away():
    lock, t0 = claimed_lock()
    # 玩家往右走出畫面（x > 2.5 就看不到了）
    leave = lambda t: [p for p in [person((t - t0) * 1.5, TAPE_Z)] if p.ground()[0] < 2.5]
    run(lock, leave, 2.0, t0)
    t1 = t0 + 2.0
    # 新的人站上膠帶，沒舉手
    stand = run(lock, lambda t: [person(0, TAPE_Z)], 2.0, t1)
    for _, f, _ in stand:
        assert f.mode != "track"
        assert f.player_index is None
    # 他舉手 -> 新的一次認領
    t2 = t1 + 2.0
    claim = run(lock, lambda t: [person(0, TAPE_Z, hands_up=True)], 1.2, t2)
    assert claim[-1][1].mode == "track"
    assert claim[-1][1].lock_id == 2


def test_other_person_cannot_claim_while_player_tracked():
    lock, t0 = claimed_lock()
    scene = lambda t: [person(0, TAPE_Z), person(0.5, TAPE_Z, hands_up=True)]
    frames = run(lock, scene, 2.0, t0)
    for _, f, _ in frames:
        assert f.lock_id == 1
        assert abs(player_x(f)) < 0.05


# ---- 遊戲控制 ----


def test_release_returns_to_idle_and_next_claim_gets_new_id():
    lock, t0 = claimed_lock()
    lock.release()
    f = lock.update([person(0, TAPE_Z)], t0)
    assert f.mode == "idle" and f.lock_id is None
    claim = run(lock, lambda t: [person(0, TAPE_Z, hands_up=True)], 1.2, t0 + 1 / 30)
    assert claim[-1][1].lock_id == 2
