#!/usr/bin/env python3
"""Drive the AURIGA car up the cone lane and stop on the ToF return from the wall.

    python demo_car.py            # interactive viewer
    python demo_car.py --headless # run 12 s and print the ToF trace

Nothing here solves Ackermann for you - the two knuckles get separate targets
from the steering trapezoid, which is the point of modelling the axle properly.
"""
import argparse
import math
import time

import mujoco
import pathlib
MODELS = pathlib.Path(__file__).resolve().parent.parent / "models"
import mujoco.viewer
import numpy as np

WB, TRACK, R = 0.260, 0.196, 0.048
STEER_LIM = 0.55


def ackermann(delta):
    """Virtual centre angle -> (left knuckle, right knuckle, turn radius)."""
    delta = float(np.clip(delta, -STEER_LIM, STEER_LIM))
    if abs(delta) < 1e-4:
        return 0.0, 0.0, math.inf
    radius = WB / math.tan(abs(delta))
    inner = math.atan(WB / (radius - TRACK / 2))
    outer = math.atan(WB / (radius + TRACK / 2))
    s = math.copysign(1.0, delta)
    left, right = (inner, outer) if s > 0 else (outer, inner)
    return s * left, s * right, radius


def main(headless):
    model = mujoco.MjModel.from_xml_path(str(MODELS / "scene_car.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)
    A = {n: aid(n) for n in ("steer_left", "steer_right", "rear_left_drive",
                             "rear_right_drive", "camera_pan", "camera_tilt")}
    tof = [model.sensor_adr[sid(n)] for n in ("tof_left", "tof_centre", "tof_right")]

    speed = 0.5

    def control(t):
        delta = 0.10 * math.sin(2 * math.pi * t / 4.0)
        left, right, _ = ackermann(delta)
        data.ctrl[A["steer_left"]] = left
        data.ctrl[A["steer_right"]] = right
        # emergency stop on the centre ToF, like the real car would.
        # A rangefinder returns -1 when the ray hits nothing, so treat only a
        # real, positive, short return as an obstacle.
        rng = data.sensordata[tof[1]]
        v = 0.0 if 0.0 <= rng < 0.30 else speed
        data.ctrl[A["rear_left_drive"]] = v / R
        data.ctrl[A["rear_right_drive"]] = v / R
        # look where the car is going
        data.ctrl[A["camera_pan"]] = 1.1 * delta
        data.ctrl[A["camera_tilt"]] = -0.12
        return delta

    if headless:
        nxt = 0.0
        while data.time < 12.0:
            delta = control(data.time)
            mujoco.mj_step(model, data)
            if data.time >= nxt:
                nxt += 1.5
                _, _, radius = ackermann(delta)
                rng = [data.sensordata[i] for i in tof]
                print(f"t={data.time:5.1f}s  delta={delta:+.2f} rad  "
                      f"R={radius:6.2f} m  "
                      f"tof L/C/R = {rng[0]:.2f} {rng[1]:.2f} {rng[2]:.2f} m  "
                      f"pos=({data.qpos[0]:+.2f}, {data.qpos[1]:+.2f})")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            t = time.time() - start
            control(t)
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - start - t)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    main(ap.parse_args().headless)
