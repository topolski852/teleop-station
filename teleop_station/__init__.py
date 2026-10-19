"""
teleop_station — runtime for the two-arm teleop station (pick and place, part inspection,
sorting).

Builds on the shared hardware layer in the ``core/`` submodule (``humanoid_core``): the CAN
daemon and its client, the limb layout, and the arm kinematics, teleop and calibration. This
package holds what only the station needs. See the README for the roadmap.
"""
