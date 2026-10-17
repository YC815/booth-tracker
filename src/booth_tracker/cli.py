"""booth-tracker 指令列：doctor / record / calibrate / eval。

macOS 開 Orbbec 要 root：sudo .venv/bin/booth-tracker <指令>（不要用 sudo uv run），見 docs/orbbec.md。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from booth_tracker.geometry import CameraPose
from booth_tracker.lock import Zone, side_by_side
from booth_tracker.paths import DATA_DIR
from booth_tracker.sources.base import FrameSource
from booth_tracker.types import Frame
from booth_tracker.venue import DEFAULT_PATH, Venue, calibrate_floor, zone_center


def _recording_path(name: str) -> Path:
    p = Path(name)
    return p if p.exists() or p.is_absolute() or "/" in name else DATA_DIR / "recordings" / name


def _source(name: str, replay: str | None) -> FrameSource:
    if name == "replay":
        from booth_tracker.sources.replay import ReplaySource

        if not replay:
            raise SystemExit("--source replay 需要 --replay 名稱或路徑")
        return ReplaySource(_recording_path(replay), loop=False, realtime=False)
    from booth_tracker.sources import make_source

    return make_source(name)


def _collect(source: FrameSource, seconds: float) -> list[Frame]:
    """收 seconds 秒（依幀的時間戳）的幀。回放讀完或即時來源等太久就提早結束。"""
    frames: list[Frame] = []
    deadline = time.monotonic() + seconds + 10
    while time.monotonic() < deadline:
        f = source.read()
        if f is None:
            if getattr(source, "name", "") == "replay":
                break
            continue
        frames.append(f)
        if f.ts - frames[0].ts >= seconds:
            break
    return frames


# ---- 指令 ----


def cmd_doctor(args: argparse.Namespace) -> int:
    from booth_tracker.doctor import run

    return run()


def cmd_record(args: argparse.Namespace) -> int:
    from booth_tracker.record import record

    out = DATA_DIR / "recordings" / args.name
    if out.exists() and not args.force:
        print(f"{out} 已存在；換名字或加 --force")
        return 1
    source = _source(args.source, None)
    source.start()
    try:
        record(source, out, args.seconds)
    finally:
        source.stop()
    return 0


def _camera_pose(args: argparse.Namespace, source: FrameSource | None) -> CameraPose | None:
    if args.height is not None:
        return CameraPose.from_angles(args.height, args.pitch or 0.0, args.roll or 0.0)
    if source is None or not source.has_depth:
        print("這個來源沒有深度，不能自動找地面；請用 --height（和 --pitch、--roll）手動指定相機姿勢")
        return None
    print(f"找地面：請讓站位區附近的地面淨空 {args.floor_seconds:g} 秒……")
    frames = [f for f in _collect(source, args.floor_seconds) if f.depth_m is not None and f.intrinsics is not None]
    if not frames:
        print("沒有收到帶深度和內參的畫面")
        return None
    pose = calibrate_floor([f.depth_m for f in frames], frames[0].intrinsics)
    if pose is None:
        print("找不到地面。檢查：地面淨空、相機往下俯一點（5–15°）、地面不要太反光；或用 --height/--pitch 手動指定")
        return None
    print(f"相機離地 {pose.height:.2f} m、俯角 {pose.pitch_deg:.1f}°、側傾 {pose.roll_deg:.1f}°（地面點 {pose.inlier_ratio:.0%}）")
    return pose


def _zone_center(args: argparse.Namespace, source: FrameSource | None, pose: CameraPose) -> tuple[float, float] | None:
    if args.zone_x is not None and args.zone_z is not None:
        return args.zone_x, args.zone_z
    if source is None:
        return None
    from booth_tracker.lift import lift
    from booth_tracker.plausible import plausible
    from booth_tracker.pose import PoseDetector

    where = "兩區中間" if args.zones == 2 else "膠帶中央"
    print(f"定站位區：請一個人站在{where} {args.person_seconds:g} 秒……")
    detector = PoseDetector(num_poses=2)
    try:
        people = []
        for f in _collect(source, args.person_seconds):
            dets = detector.detect(f.rgb, int(f.ts * 1000))
            people.append([p for p in (lift(d, f, pose) for d in dets) if plausible(p)])
    finally:
        detector.close()
    center = zone_center(people)
    if center is None:
        print("站位區：大部分時間都沒看到人，請確認有人站在畫面裡，或用 --zone-x/--zone-z 手動指定")
    return center


def cmd_calibrate(args: argparse.Namespace) -> int:
    need_source = args.height is None or args.zone_x is None or args.zone_z is None
    source = _source(args.source, args.replay) if need_source else None
    if source is not None:
        source.start()
    try:
        pose = _camera_pose(args, source)
        if pose is None:
            return 1
        center = _zone_center(args, source, pose)
        if center is None:
            return 1
    finally:
        if source is not None:
            source.stop()
    cx, cz = center
    if args.zones == 2:
        zones = side_by_side(cx, cz, args.radius or 0.5, args.gap)
    else:
        r = args.radius or 0.7
        zones = [Zone(cx, cz, r, r + 0.3)]
    path = Venue(camera=pose, zones=zones).save(args.out)
    for k, z in enumerate(zones):
        print(f"站位區 {k}：中心 x={z.x:+.2f} m、距相機 {z.z:.2f} m、半徑 {z.r_in:.2f} m")
    print(f"已存 {path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from booth_tracker.evaluate import annotate, evaluate
    from booth_tracker.sources.replay import ReplaySource
    from booth_tracker.tracker import BoothTracker

    venue = Venue.load(args.venue)
    source = ReplaySource(_recording_path(args.replay), loop=False, realtime=False)
    writer = None

    def on_frame(tf) -> None:
        nonlocal writer
        img = annotate(tf, venue)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), source._fps, (w, h))
        writer.write(img)

    tracker = BoothTracker(source, venue)
    with tracker:
        report = evaluate(tracker, on_frame=on_frame if args.video else None)
    if writer is not None:
        writer.release()
        print(f"標註影片：{args.video}")
    print(report.text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="booth-tracker", description="體感攤位遊戲的玩家追蹤")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Orbbec 相機自檢（macOS 要 sudo）").set_defaults(func=cmd_doctor)

    p = sub.add_parser("record", help="錄原始 RGB-D 到資料目錄的 recordings/<name>")
    p.add_argument("--source", choices=["orbbec", "webcam"], default="orbbec")
    p.add_argument("--name", required=True)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--force", action="store_true", help="覆蓋同名錄影")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("calibrate", help="換場地時跑：找地面、定站位區，存 venue.json")
    p.add_argument("--source", choices=["orbbec", "webcam", "replay"], default="orbbec")
    p.add_argument("--replay", help="--source replay 時的錄影名稱或路徑")
    p.add_argument("--zones", type=int, choices=[1, 2], default=1)
    p.add_argument("--radius", type=float, help="站位區半徑（預設單區 0.7、雙區 0.5）")
    p.add_argument("--gap", type=float, default=0.4, help="雙區中間的空白帶寬度")
    p.add_argument("--zone-x", type=float, help="手動指定站位區中心（世界座標 x）")
    p.add_argument("--zone-z", type=float, help="手動指定站位區中心（離相機水平距離）")
    p.add_argument("--height", type=float, help="手動指定相機離地高度（不自動找地面）")
    p.add_argument("--pitch", type=float, help="手動俯角，往下為正")
    p.add_argument("--roll", type=float, help="手動側傾")
    p.add_argument("--floor-seconds", type=float, default=2.0)
    p.add_argument("--person-seconds", type=float, default=3.0)
    p.add_argument("--out", type=Path, default=DEFAULT_PATH)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("eval", help="用錄影跑整條管線，列出鎖定事件")
    p.add_argument("--replay", required=True, help="錄影名稱或路徑")
    p.add_argument("--venue", type=Path, default=DEFAULT_PATH)
    p.add_argument("--video", type=Path, help="輸出標註影片 .mp4")
    p.set_defaults(func=cmd_eval)
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
