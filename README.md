# teleop-station

Runtime for a two-arm teleop station built from Berkeley Humanoid Lite arms. It is aimed at
pick and place, part inspection and sorting. An operator drives the arms from a VR headset,
and a manufacturer can program them like an industrial arm: teach points, then run programs
that move through them.

**Status: skeleton.** The repo layout and the shared hardware layer are in place. The station's
own runtime is not written yet. Until milestone 1 lands, the working Quest arm teleop still
runs from [`humanoid-control`](https://github.com/topolski852/humanoid-control) on the arms PC
(set up by its `deploy/convert-to-arms-teleop.sh`).

## Layout

- `core/`: the [`humanoid-core`](https://github.com/topolski852/humanoid-core) **git
  submodule**, shared with humanoid-control. It holds the CAN daemon (`core/daemon/`), the
  daemon client, the limb layout and the arm math (kinematics, teleop, calibration, operator
  profiles). `humanoid_core` at the repo root is a symlink into it.
- `teleop_station/`: what only the station needs.
  - `vision/`: the camera as a measuring instrument for the accuracy test (roadmap 2).
    AprilTag poses, with every tag reported in a reference frame fitted to the tags that
    stay still during a test (table and torso), and averaging over the frames at each stop.
- `scripts/`: `make_targets.py` (print sheets: tags and the ChArUco board),
  `calibrate_camera.py` (camera intrinsics), `track_tags.py` (live wrist positions and the
  camera's noise floor).
- `tests/`: offline tests, no robot needed. Run each one with `python tests/test_<name>.py`.

```bash
git clone --recurse-submodules https://github.com/topolski852/teleop-station.git
cd teleop-station
(cd core/daemon && make)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tests/test_core_link.py
```

### Setting up the camera

Nothing here moves an arm. The tag set (family, ids, names, sizes, which tags are the
reference) is in `config/tags.json`: 25h9 tags, ids 0-2 on the right arm (hand, wrist
actuator, shoulder_roll), 3-5 on the left arm, 6-7 on the torso (top, front), 8-9 on the
table. Tags 6-9 are the reference.

1. `python scripts/make_targets.py`, then print `targets/charuco.pdf` (and `targets/tags.pdf`
   for spare tags) at **100% / actual size** on matte paper, and mount it flat and rigid.
   Check the 100 mm bar. If the printer scaled the page, the board still works, but measure
   it: span 4-5 squares with calipers in both directions and divide. Measure each tag's
   **black square** (not the cut-out paper) and put it in `config/tags.json`.
2. Fix the camera and lock it: manual focus, fixed zoom, manual exposure. On a webcam,
   pass `--exposure` (the live view's `[` and `]` keys find a value; auto exposure clips the
   white and breaks decoding). Through an HDMI capture card, set them on the camera body.
3. `python scripts/calibrate_camera.py --source /dev/video0 --square <m> --width W --height H
   --out calib/<camera>.json`. Aim for RMS under 0.5 px. Redo it if focus or zoom ever moves.
4. Once the table and torso are where they'll stay, survey the reference tags:
   `python scripts/track_tags.py --source /dev/video0 --calib calib/<camera>.json
   --exposure <n> --survey 90` (writes `config/layout.json`, not committed). Redo it whenever
   either one has moved.
5. `python scripts/track_tags.py --source /dev/video0 --calib calib/<camera>.json --exposure <n>`
   shows every tag live. With nothing moving, the `±` spread is the single-frame noise; the
   noise floor of a stop's averaged reading is lower. `--measure N` prints one N-frame reading
   as JSON.

The Logitech C310 (1280x960) calibrated at 0.28 px RMS. With exposure locked, a 30-frame
reading of a still arm tag repeats to 0.12-0.19 mm, with 0.1-0.6 mm of drift over 2 minutes
(possibly the webcam warming up). Its field of view only just covers the shoulders, so arm
poses that reach outward leave the frame.

A fix to the daemon or the arm math goes into humanoid-core, not here. Then move the `core/`
pointer forward in this repo (see core's README).

## The hardware, as it is today

- **Two 5-joint arms**: shoulder pitch, roll and yaw, elbow pitch, wrist yaw. The IK controls
  where the **wrist pivot** is, not the hand's orientation. The last joint is an inline twist
  that barely moves the wrist. Top-down picks work; tilting a part toward a camera is limited.
- **Reach is about 0.19–0.33 m** from each shoulder (`tests/test_core_link.py` prints it).
  That sets the station's working envelope.
- **No gripper yet.** The design is on hold, so nothing here should assume a specific one.
- **No legs and no IMU.** The station layout is `left_arm` + `right_arm` only.

Keep the kinematics driven by the URDF geometry in core, so different arms can replace these
without a rewrite.

## Roadmap

1. **Arm-only service.** A station control service and web server for two arms: connect, arm,
   E-stop, deadman, per-arm rest. Port the Quest WebXR bridge (`humanoid_control/web/xr.py`
   and the `app/xr` page) from humanoid-control. If both repos then use the bridge the same
   way, move it into core instead of keeping two copies.
2. **Accuracy test.** Measure repeatability (returning to a taught point) and accuracy
   (reaching a computed point). Fix a camera, put an AprilTag on each wrist, cycle each arm
   through 5–8 taught poses 50–100 times, and log the commanded, encoder and camera positions
   at each stop. Report the spread per point and the drift over time.
   The numbers decide whether pick and place needs camera correction.
3. **Motion core.** Named points, `MoveJ` with planned speed profiles, speed and acceleration
   limits, user frames (points taught relative to a fixture), and both arms finishing moves
   together. Include a dry-run mode that simulates moves without driving the motors.
4. **Programs.** Programs stored as data (move, wait, set/read I/O, loop, branch), a runner,
   and a Python API for integrators. Teach points from the headset: teleop to a pose and
   press a button to save it.
5. **Teach pendant UI.** A web page to edit points and programs and to step through and run
   them.
6. **Integration.** Digital I/O, then Modbus TCP, so a cell PLC can run the station.
