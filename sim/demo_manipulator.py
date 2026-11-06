#!/usr/bin/env python3
"""Drive the AURIGA mecanum base in a square while the arm works through poses.

    python demo_manipulator.py            # interactive viewer
    python demo_manipulator.py --headless # run 20 s and print the odometry

The base command is a plain (vx, vy, wz) twist; the wheel rates come from the
inverse mecanum kinematics below and the sideways motion is produced by the
roller contacts, not by a plugin.
"""
import argparse
import time

import mujoco
import pathlib
MODELS = pathlib.Path(__file__).resolve().parent.parent / "models"
import mujoco.viewer
import numpy as np

R = 0.065                       # wheel radius
LXY = 0.180 + 0.1975            # (lx + ly) for the mecanum mixer

WHEELS = ["front_left_wheel", "front_right_wheel",
          "rear_left_wheel", "rear_right_wheel"]

# (vx, vy, wz, duration) - a square driven without ever turning, then a spin
LEG = 3.0
ROUTE = [(0.35, 0.0, 0.0, LEG), (0.0, 0.35, 0.0, LEG),
         (-0.35, 0.0, 0.0, LEG), (0.0, -0.35, 0.0, LEG),
         (0.0, 0.0, 1.2, 2.6), (0.0, 0.0, 0.0, 1.0)]

# (yaw, shoulder, elbow, wrist, gripper)
POSES = [(0.0, -0.95, 1.15, 0.62, 0.018),
         (0.6, -0.35, 0.45, 0.35, 0.030),
         (0.0, -1.45, 2.35, 0.55, 0.004)]


def mecanum(vx, vy, wz):
    """Body twist -> the four wheel rates, in the order of WHEELS."""
    return np.array([(vx - vy - LXY * wz) / R,
                     (vx + vy + LXY * wz) / R,
                     (vx + vy - LXY * wz) / R,
                     (vx - vy + LXY * wz) / R])


def main(headless):
    model = mujoco.MjModel.from_xml_path(str(MODELS / "scene_manipulator.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    wheel_act = [aid(w) for w in WHEELS]
    arm_act = [aid(n) for n in
               ("arm_yaw", "shoulder_pitch", "elbow", "wrist_pitch", "gripper")]

    total = sum(leg[3] for leg in ROUTE)

    def control(t):
        # base: walk through the route
        acc, cmd = 0.0, (0.0, 0.0, 0.0)
        for vx, vy, wz, dur in ROUTE:
            if t < acc + dur:
                cmd = (vx, vy, wz)
                break
            acc += dur
        data.ctrl[wheel_act] = mecanum(*cmd)
        # arm: hold each pose for a third of the route
        pose = POSES[min(int(t / (total / len(POSES))), len(POSES) - 1)]
        data.ctrl[arm_act] = pose

    if headless:
        while data.time < total:
            control(data.time)
            mujoco.mj_step(model, data)
        print(f"t={data.time:.1f}s  base xy = "
              f"({data.qpos[0]:+.3f}, {data.qpos[1]:+.3f}) m   "
              f"yaw = {2 * np.arctan2(data.qpos[6], data.qpos[3]):+.2f} rad")
        print("square closes on itself if the mecanum mixer is right")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            t = time.time() - start
            control(t % total)
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - start - t)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    main(ap.parse_args().headless)
