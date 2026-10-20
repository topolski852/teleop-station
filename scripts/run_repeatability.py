#!/usr/bin/env python3
"""Repeatability run: cycle the arms through taught poses and measure every stop.

    python scripts/run_repeatability.py --poses runs/poses_demo.json --dry-run   # moves nothing
    sudo systemctl stop humanoid-web                                              # then, for real:
    python scripts/run_repeatability.py --poses runs/poses_demo.json --cycles 50

Each cycle visits every pose (left pose i and right pose i together, or one arm with
``--arms``). Every pose is approached the same way: a synchronised minimum-jerk move to a
point ``--approach`` degrees beyond it, then a slow final move in, so backlash is taken up
from the same side every time. At each stop: settle, then the commanded angles, the encoder
angles and a ``--shots``-shot camera reading of every tag are logged (``runs/rep_<time>.jsonl``).
``scripts/report_repeatability.py`` turns that into spread per pose and drift.

The whole program is planned and checked (limits, speed, acceleration) before anything is
enabled. ``--dry-run`` does exactly that and then runs the first cycle against a simulated
arm in real time. Nothing is sent to the daemon; the camera is used only if ``--camera`` is
given.

A real run asks you to type ``yes`` with the arms clear before it enables anything. Ctrl-C
stops and holds; you then choose to return to rest or to E-stop. A second Ctrl-C, any
safety check (tracking error, stale telemetry, a joint fault) and any crash E-stop: the
daemon puts every joint in DAMPING. At the end the arms go back to rest, then DAMPING, then
IDLE.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from teleop_station.motion import (DaemonBus, Executor, JointLimits, MotionAborted,   # noqa: E402
                                   MotionStopped, SimBus, approach_move, movej)

DEG = np.pi / 180
DAEMON_CONFIG = "/home/nse/humanoid-studio/configs/humanoid_arms.json"


def build_program(poses: dict, arms, approach, vmax, amax, final_vmax, limits):
    """The stops of one cycle: names, rest, [(label, target, direction)], and the checked
    trajectories of a whole cycle (pose_last -> pose_0 ... and rest -> pose_0 for the start)."""
    names = [n for side in arms for n in poses["joints"][side]]
    rest = np.array([poses["rest"][n] for n in names])
    lists = [poses["poses"][side] for side in arms]
    if any(len(l) == 0 for l in lists):
        raise SystemExit(f"no taught poses for: {[s for s, l in zip(arms, lists) if not l]}")
    n_stops = max(len(l) for l in lists)
    stops = []
    for i in range(n_stops):
        q, label = {}, []
        for side, l in zip(arms, lists):
            p = l[i % len(l)]
            q.update(p["q"]); label.append(p["name"])
        target = np.array([q[n] for n in names])
        # Approach from +approach on every joint, unless that leaves the limits: then from -.
        # Fixed per pose, so the same side is used every cycle.
        direction = np.array([1.0 if target[k] + approach <= limits.upper[n] else -1.0
                              for k, n in enumerate(names)])
        stops.append(("+".join(label), target, direction))
    problems = []
    def plan(q_from, stop):
        tr = approach_move(names, q_from, stop[1], approach, stop[2], vmax, amax, final_vmax)
        problems.extend(f"move to {stop[0]}: {p}" for p in limits.check(tr, vmax, amax))
        return tr
    first = plan(rest, stops[0])
    cycle = [plan(stops[k - 1][1], stops[k]) for k in range(1, len(stops))]
    wrap = plan(stops[-1][1], stops[0])
    home = movej(names, stops[-1][1], rest, vmax, amax)
    problems.extend(f"move to rest: {p}" for p in limits.check(home, vmax, amax))
    for label, target, _ in stops:
        problems.extend(f"pose {label}: {v}" for v in limits.violations(names, target))
    return names, rest, stops, first, cycle, wrap, home, problems


class Camera:
    """The Sony (or any source) measuring every tag in the station reference frame."""

    def __init__(self, source, calib_path, survey_shots, log):
        from track_tags import DEFAULT_LAYOUT_FILE, DEFAULT_TAGS_FILE, load_layout
        from teleop_station.vision import CameraCalibration, TagSet, TagTracker, survey
        from teleop_station.vision.camera import describe, open_device
        self.calib = CameraCalibration.load(calib_path)
        tags = TagSet.load(DEFAULT_TAGS_FILE)
        self.cap = open_device(source, *self.calib.image_size)
        log("camera: " + describe(self.cap))
        # Survey the reference tags now: the table and torso can shift during setup.
        seen = []
        for _ in range(survey_shots):
            ok, f = self.cap.read()
            if ok:
                seen.append(TagTracker(self.calib, tags).detect(f))
        layout = survey(tags, seen)
        missing = set(tags.reference[1:]) - set(layout)
        if missing:
            log(f"camera: reference tags {sorted(missing)} not surveyed; using the old layout for them")
            old = load_layout(DEFAULT_LAYOUT_FILE)
            layout.update({i: old[i] for i in missing if i in old})
        self.tags = replace(tags, layout=layout)
        self.tracker = TagTracker(self.calib, self.tags)
        self.layout_json = {str(i): p.to_json() for i, p in layout.items()}

    def reading(self, shots, on_shot=None) -> dict:
        from teleop_station.vision import mean_pose
        per, res = {}, []
        for _ in range(shots):
            ok, f = self.cap.read()
            if on_shot:
                on_shot()
            if not ok:
                continue
            rel, fit = self.tracker.in_reference(f)
            if fit:
                res.append(max(fit.per_tag_px.values()))
            for n, p in rel.items():
                per.setdefault(n, []).append(p)
        out = {}
        for n, ps in per.items():
            m, sd = mean_pose(ps)
            out[n] = {"pos_mm": (m.t * 1000).round(4).tolist(), "std_mm": (sd * 1000).round(4).tolist(),
                      "R": m.R.round(6).tolist(), "n": len(ps)}
        return {"tags": out, "ref_px_max": round(max(res), 3) if res else None, "shots": shots}

    def close(self):
        self.cap.release()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--cycles", type=int, default=50)
    ap.add_argument("--arms", default="both", choices=("both", "left", "right"))
    ap.add_argument("--vmax", type=float, default=15.0, help="deg/s per joint")
    ap.add_argument("--amax", type=float, default=60.0, help="deg/s^2 per joint")
    ap.add_argument("--approach", type=float, default=3.0, help="deg beyond each pose, then in")
    ap.add_argument("--final-vmax", type=float, default=5.0, help="deg/s for the final approach")
    ap.add_argument("--settle", type=float, default=1.0, help="extra seconds after settling")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--camera", default="gphoto2:runs/shots_rep_{time}", help="camera source")
    ap.add_argument("--calib", default=str(ROOT / "calib" / "a7s3_sigma16_2768x1848.json"))
    ap.add_argument("--survey-shots", type=int, default=10)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="plan, check, simulate one cycle; send nothing")
    ap.add_argument("--limits", default="soft", choices=("soft", "hard"),
                    help="soft: the URDF's range (teleop's); hard: the ESC firmware's wider range")
    ap.add_argument("--config", default=DAEMON_CONFIG)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%dT%H%M%S")
    poses = json.loads(Path(a.poses).read_text())
    if not poses.get("rest"):
        sys.exit("the poses file has no rest pose (teach_poses.py: 'rest')")
    arms = ["left", "right"] if a.arms == "both" else [a.arms]
    limits = JointLimits.for_arms(a.config, arms=[f"{s}_arm" for s in arms], tier=a.limits)
    vmax, amax, final_vmax, approach = a.vmax * DEG, a.amax * DEG, a.final_vmax * DEG, a.approach * DEG
    names, rest, stops, first, cycle, wrap, home, problems = build_program(
        poses, arms, approach, vmax, amax, final_vmax, limits)

    move_t = first.duration + sum(t.duration for t in cycle)
    per_stop = 2.0 + a.settle + (0 if a.no_camera else a.shots * 2.1)
    cycle_t = sum(t.duration for t in cycle) + wrap.duration + len(stops) * per_stop
    print(f"program: {len(stops)} stops per cycle ({', '.join(s[0] for s in stops)}), arms {arms}")
    print(f"  moves {vmax / DEG:.0f} deg/s, final approach {a.approach:.0f} deg at {a.final_vmax:.0f} deg/s")
    for label, target, d in stops:
        print(f"  {label:8s} " + "  ".join(f"{n.replace('_joint', '').replace('shoulder_', 'sh_')} {v / DEG:+6.1f}"
                                         f"{'+' if dd > 0 else '-'}" for n, v, dd in zip(names, target, d)))
    print(f"  one cycle ~{cycle_t / 60:.1f} min (moves {move_t:.0f} s + {per_stop:.0f} s per stop); "
          f"{a.cycles} cycles ~{(first.duration + a.cycles * cycle_t + home.duration) / 60:.0f} min")
    if problems:
        print("\nREJECTED, nothing was enabled:\n  " + "\n  ".join(problems))
        return 1
    print("  every move checked: limits, speed, acceleration OK")

    out_path = Path(a.out or ROOT / "runs" / f"rep_{'dry_' if a.dry_run else ''}{stamp}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = out_path.open("w")
    def log(msg):
        print(msg, flush=True)
        out.write(json.dumps({"kind": "event", "t": round(time.time(), 3), "msg": msg}) + "\n"); out.flush()

    cam = None
    if not a.no_camera and not (a.dry_run and a.camera == ap.get_default("camera")):
        cam = Camera(a.camera.replace("{time}", stamp), a.calib, a.survey_shots, log)
    if a.dry_run:
        bus = SimBus({n: v for n, v in zip(names, rest)})
        cycles = 1
        log("DRY RUN: simulated arm, nothing is sent to the daemon")
    else:
        print(f"\nThis will ENABLE and MOVE the {' and '.join(arms)} arm(s) for {a.cycles} cycles.")
        if input("Are the arms and the space around them clear? Type 'yes' to start: ").strip() != "yes":
            print("not started")
            return 1
        bus = DaemonBus(a.config)
        cycles = a.cycles

    out.write(json.dumps({"kind": "header", "started": stamp, "dry_run": a.dry_run, "args": vars(a),
                          "names": names, "rest": rest.tolist(),
                          "stops": [{"label": l, "target": t.tolist(), "direction": d.tolist()} for l, t, d in stops],
                          "limits": {n: [limits.lower[n], limits.upper[n]] for n in names},
                          "camera_layout": cam.layout_json if cam else None}) + "\n")
    ex = Executor(bus, names, limits, log=log)

    presses = {"n": 0}
    def on_sigint(signum, frame):
        presses["n"] += 1
        if presses["n"] == 1:
            print("\n[stop requested: finishing this tick and holding. Ctrl-C again = E-STOP]")
            ex.request_stop()
        else:
            ex.estop("second Ctrl-C")
            raise KeyboardInterrupt
    signal.signal(signal.SIGINT, on_sigint)

    def measure(cycle_i, k, label, target, settle):
        enc = []
        def sample():
            s = bus.read(names); enc.append([s[n].position for n in names])
        if a.settle:
            t_end = time.time() + a.settle
            while time.time() < t_end:
                sample(); time.sleep(0.05)
        cam_r = cam.reading(a.shots, sample) if cam else None
        if not enc:
            sample()
        E = np.array(enc)
        rec = {"kind": "stop", "t": round(time.time(), 3), "cycle": cycle_i, "stop": k, "label": label,
               "commanded": target.tolist(), "encoder_mean": E.mean(0).tolist(), "encoder_std": E.std(0).tolist(),
               "settle": settle, "camera": cam_r}
        out.write(json.dumps(rec) + "\n"); out.flush()
        err = np.degrees(E.mean(0) - target)
        tags = "" if not cam_r else f", {len(cam_r['tags'])} tags"
        print(f"  cycle {cycle_i + 1}/{cycles} {label:8s} settled {settle['settled']} in {settle['t']:.1f} s, "
              f"encoder - command max {np.abs(err).max():.2f} deg{tags}", flush=True)

    status = "ok"
    try:
        q0 = ex.enable()
        start = approach_move(names, q0, stops[0][1], approach, stops[0][2], vmax, amax, final_vmax)
        for c in range(cycles):
            for k, (label, target, d) in enumerate(stops):
                tr = start if (c == 0 and k == 0) else (wrap if k == 0 else cycle[k - 1])
                ex.run(tr, vmax, amax)
                measure(c, k, label, target, ex.settle())
        ex.run(movej(names, ex.last_target, rest, vmax, amax), vmax, amax)
        ex.settle()
        ex.release()
    except MotionStopped:
        status = "stopped"
        signal.signal(signal.SIGINT, lambda *_: (ex.estop("Ctrl-C at the prompt"), sys.exit(1)))
        ans = input("Stopped and holding. [r] return to rest slowly and release, anything else = E-stop: ").strip()
        if ans == "r":
            ex.run(movej(names, ex.last_target, rest, vmax / 2, amax), vmax, amax)
            ex.settle()
            ex.release()
        else:
            ex.estop("operator chose E-stop after a stop")
    except MotionAborted as e:
        status = f"aborted: {e}"
    except KeyboardInterrupt:
        status = "E-stopped (second Ctrl-C)"
    except BaseException as e:
        ex.estop(f"crash: {type(e).__name__}: {e}")
        status = f"crashed: {e}"
        raise
    finally:
        log(f"run ended: {status}")
        out.close()
        if cam:
            cam.close()
        bus.close()
        print(f"log: {out_path}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
