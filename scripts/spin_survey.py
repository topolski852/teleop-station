#!/usr/bin/env python3
"""Survey the hand cubes (and forearm tag pairs) by turning the wrists in steps.

    python scripts/spin_survey.py --path runs/path_raw_test3.mirror.planned.json --dry-run
    python scripts/spin_survey.py --path runs/path_raw_test3.mirror.planned.json

The arms go out along the path's recorded lead-in to its first stop (S1), clear of the table.
There only the wrists turn, from ``-range`` to ``+range`` degrees about where they started,
in ``--step`` degree steps; after each step the arm settles and the camera takes ``--shots``
stills. Turning the cube past the camera shows its side faces in neighbouring pairs, which is
what ``survey_body`` needs to place every face, the far ones included. Then the wrists turn
back, the arms return along the lead-out, and each body's layout is written to
``config/body_layout.json`` (with how each tag was reached and from how many frames).

The wrist sweep needs wrist limits at least as wide as the range (``scripts/apply_arm_config.py``
after widening them in the daemon config). Everything else is the runner's: every move is
planned and checked first, a real run asks for 'yes', and any tracking, telemetry or fault
problem E-stops. Ctrl-C here E-stops too (the arms are only ever a wrist turn from a stop).
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

from run_repeatability import DAEMON_CONFIG, build_path_program                    # noqa: E402
from teleop_station.motion import (DaemonBus, Executor, JointLimits, MotionAborted,  # noqa: E402
                                   MotionStopped, SimBus, movej)

DEG = np.pi / 180


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", help="planned path with a lead-in (edit_path.py --mirror/--combine)")
    ap.add_argument("--at-rest", action="store_true",
                    help="don't move the arms: enable ONLY the wrists and turn them where the arms rest "
                         "(when the cubes clear the table there)")
    ap.add_argument("--arms", default="both", choices=("both", "left", "right"))
    ap.add_argument("--step", type=float, default=45.0, help="wrist step, deg")
    ap.add_argument("--range", type=float, default=180.0, help="turn each wrist this far each way, deg")
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--vmax", type=float, default=15.0)
    ap.add_argument("--amax", type=float, default=60.0)
    ap.add_argument("--limits", default="hard", choices=("soft", "hard"))
    ap.add_argument("--camera", default=None, help="camera source (default gphoto2, keeping the shots)")
    ap.add_argument("--calib", default=str(ROOT / "calib" / "a7s3_sigma16_2768x1848.json"))
    ap.add_argument("--out-layout", default=str(ROOT / "config" / "body_layout.json"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default=DAEMON_CONFIG)
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%dT%H%M%S")
    arms = ["left", "right"] if a.arms == "both" else [a.arms]
    limits = JointLimits.for_arms(a.config, arms=[f"{s}_arm" for s in arms], tier=a.limits)
    vmax, amax = a.vmax * DEG, a.amax * DEG
    if a.at_rest:
        # Only the wrists: every other joint stays limp, resting on the table as it is now.
        names = [f"{s}_wrist_yaw_joint" for s in arms]
        lead_in = lead_out = []
        problems = []
        if a.dry_run:
            rest = np.zeros(len(names))
        else:
            probe = DaemonBus(a.config)
            st = probe.read(names); probe.close()
            rest = np.array([st[n].position for n in names])
    else:
        if not a.path:
            sys.exit("give --path (a planned path with a lead-in) or --at-rest")
        names, rest, stops, steps, problems, plan, lead_in, lead_out = build_path_program(a.path, arms, vmax, amax, limits)
        if not lead_in:
            sys.exit("the path has no lead-in; make it with edit_path.py --combine or --mirror")
    # The rest of the taught path isn't run here; only its lead-in and lead-out are.
    problems = [p for p in problems if "of each other" not in p]
    s1 = lead_in[-1].q[-1].copy() if lead_in else rest.copy()
    wrist = [k for k, n in enumerate(names) if n.endswith("wrist_yaw_joint")]
    # One monotonic sweep: out to -range, then up in steps to +range (shots at each), then home.
    order = list(np.arange(-a.range, a.range + 1e-6, a.step))
    targets = []
    for o in order:
        q = s1.copy(); q[wrist] = s1[wrist] + o * DEG
        targets.append((o, q))
    sweep, prev = [], s1
    for o, q in targets:
        sweep.append((o, movej(names, prev, q, vmax, amax), True)); prev = q
    sweep.append((0.0, movej(names, prev, s1, vmax, amax), False))
    for o, tr, _ in sweep:
        problems.extend(f"wrist to {o:+.0f}: {p}" for p in limits.check(tr, vmax, amax))
    if len(arms) == 2 and not a.at_rest:
        from teleop_station.motion.path import arm_clearance
        d, k, at = arm_clearance(names, lead_in + [tr for _, tr, _ in sweep] + lead_out)
        if d < 0.15:
            problems.append(f"the arms come within {d * 1000:.0f} mm of each other during the survey's moves")
        else:
            print(f"  arms stay {d * 1000:.0f} mm apart (centre-lines) through the lead-in, sweep and lead-out")
    where = ("arms resting where they are, ONLY the wrists enabled" if a.at_rest else
             f"arm(s) out to S1 via the recorded lead-in ({lead_in[0].duration:.0f} s)")
    print(f"spin survey: {' and '.join(arms)}: {where}; wrists {order[0]:+.0f}..{order[-1]:+.0f} deg about "
          f"{', '.join(f'{np.degrees(s1[k]):+.0f}' for k in wrist)} in {a.step:.0f} deg steps "
          f"({len(targets)} positions x {a.shots} shots), then back")
    if problems:
        print("REJECTED, nothing was enabled:\n  " + "\n  ".join(problems))
        print("(if it's the wrist limits: widen wrist_yaw position_limits in the daemon config, "
              "then scripts/apply_arm_config.py)")
        return 1
    print("  every move checked: limits, speed, acceleration OK")

    cam = tracker = None
    source = a.camera or (None if a.dry_run else f"gphoto2:runs/shots_spin_{stamp}")
    if source:
        from teleop_station.vision import CameraCalibration, TagSet, TagTracker
        from teleop_station.vision.camera import describe, open_device
        calib = CameraCalibration.load(a.calib)
        tags = TagSet.load(ROOT / "config" / "tags.json")
        tracker = TagTracker(calib, replace(tags, body_layout={}))
        cam = open_device(source, *calib.image_size)
        print("camera:", describe(cam))
    if a.dry_run:
        bus = SimBus({n: v for n, v in zip(names, rest)})
        print("DRY RUN: simulated arm, nothing is sent to the daemon")
    else:
        if input("This will ENABLE and MOVE the arm(s). Arms and the space around them clear? Type 'yes': ").strip() != "yes":
            print("not started")
            return 1
        bus = DaemonBus(a.config)
    ex = Executor(bus, names, limits, log=print)
    signal.signal(signal.SIGINT, lambda *_: (ex.estop("Ctrl-C"), (_ for _ in ()).throw(KeyboardInterrupt)))
    sightings, log_rows = [], []
    status = "ok"
    try:
        q0 = ex.enable()
        if lead_in:
            tr = movej(names, q0, lead_in[0].q[0], vmax, amax)
            for seg in lead_in:
                tr = tr + seg
            ex.run(tr, vmax, amax); ex.settle()
        elif np.abs(q0 - s1).max() > np.radians(1.0):
            raise MotionAborted("the wrists moved between planning and enabling; run it again")
        else:
            ex.last_target = s1.copy()
        for o, tr, shoot in sweep:
            ex.run(tr, vmax, amax)
            st = ex.settle()
            if cam is not None and shoot:
                for _ in range(a.shots):
                    ok, f = cam.read()
                    if ok:
                        seen = tracker.detect(f)
                        sightings.append(seen)
                        log_rows.append({"wrist_offset_deg": o, "seen": sorted(seen)})
                print(f"  wrist {o:+5.0f} deg: settled {st['settled']}, tags seen {log_rows[-1]['seen'] if log_rows else '-'}")
            else:
                print(f"  wrist {o:+5.0f} deg: settled {st['settled']}")
        for seg in lead_out:
            ex.run(seg, vmax, amax)
        if np.abs(ex.last_target - rest).max() > 1e-6:
            ex.run(movej(names, ex.last_target, rest, vmax, amax), vmax, amax)
        ex.settle()
        ex.release()
    except (MotionAborted, MotionStopped) as e:
        status = f"aborted: {e}"
        if not ex.estopped:
            ex.estop(str(e))
    except KeyboardInterrupt:
        status = "E-stopped (Ctrl-C)"
    except BaseException as e:
        ex.estop(f"crash: {type(e).__name__}: {e}")
        raise
    finally:
        if cam is not None:
            cam.close()
        bus.close()
        print(f"run ended: {status}")
    if not sightings:
        return 0 if status == "ok" else 1

    from teleop_station.vision import TagSet, save_body_layout, survey_body
    tags = TagSet.load(ROOT / "config" / "tags.json")
    layout, info = {}, {}
    for body, ids in tags.bodies.items():
        if len(ids) < 2 or not any(body.startswith(s) for s in arms):
            continue
        lay, inf = survey_body(ids, sightings, min_pairs=2)
        layout[body] = lay
        pairs = {f"{x}-{y}": sum(x in s and y in s for s in sightings) for i, x in enumerate(ids) for y in ids[i + 1:]}
        info[body] = {"reached": {str(k): v for k, v in inf.items()}, "pairs_seen": pairs,
                      "missing": [i for i in ids[1:] if i not in lay]}
        print(f"{body}: anchor {ids[0]}, placed {sorted(lay)}" +
              (f", NOT placed {info[body]['missing']}" if info[body]["missing"] else "") +
              f"; pairs seen together {pairs}")
    old = Path(a.out_layout)
    merged = {}
    if old.exists():
        from teleop_station.vision import load_body_layout
        merged = load_body_layout(old)
    merged.update(layout)
    save_body_layout(a.out_layout, merged, {"surveyed": stamp, "source": "spin_survey", **info})
    Path(ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / f"spin_{stamp}.json").write_text(json.dumps({"rows": log_rows, "info": info}, indent=2) + "\n")
    print(f"wrote {a.out_layout}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
