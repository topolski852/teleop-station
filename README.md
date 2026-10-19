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
- `teleop_station/`: what only the station needs (empty for now).
- `tests/`: offline tests, no robot needed. Run each one with `python tests/test_<name>.py`.

```bash
git clone --recurse-submodules https://github.com/topolski852/teleop-station.git
cd teleop-station
(cd core/daemon && make)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tests/test_core_link.py
```

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
