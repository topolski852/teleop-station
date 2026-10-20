"""
Joint-space motion for the station: plan a whole move ahead of time, check it, then stream it.

Every move is a dense, precomputed joint trajectory (``trajectory``), validated against the
joint limits and the speed and acceleration caps (``limits``) before a single command is
sent. One ``Executor`` streams any trajectory to a ``JointBus`` (``bus``: the real daemon, or
a simulated arm for dry runs) and watches it while it does: tracking error, stale telemetry,
joint faults. Anything wrong is an E-stop (the daemon damps every joint).

Angles are DEVICE frame radians throughout (what telemetry reports and ``SET_POSITION``
takes). Cartesian moves (MoveL, pivoting about the tool point) will produce trajectories in
the same form and go through the same checks and executor.
"""
from .bus import DaemonBus, JointBus, JointSample, SimBus
from .executor import Executor, MotionAborted, MotionStopped
from .limits import JointLimits
from .path import PlannedPath, RawPath, combine_arms, find_dwells, plan_path
from .trajectory import Trajectory, approach_move, movej

__all__ = [
    "DaemonBus", "JointBus", "JointSample", "SimBus",
    "Executor", "MotionAborted", "MotionStopped",
    "JointLimits", "Trajectory", "approach_move", "movej",
    "PlannedPath", "RawPath", "combine_arms", "find_dwells", "plan_path",
]
