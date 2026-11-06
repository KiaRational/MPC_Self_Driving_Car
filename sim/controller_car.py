#!/usr/bin/env python3
"""Interactive keyboard controller for the AURIGA car on the figure-8 track.

    python controller_car.py                # WASD drive
    python controller_car.py --autopilot     # start in scripted-lap mode

IMPORTANT: click the "AURIGA dashboard" window (not the 3D viewer) before
typing. MuJoCo's 3D viewer hard-wires W/A/S/... to its own camera/render
shortcuts (W = wireframe, and friends) no matter what - it will never see
these as drive keys, so it is view-only here. Once the dashboard window has
focus:

    W / S       speed up / slow down (reverse below zero)
    A / D       steer left / right
    I / K       mast camera tilt up / down   (manual, independent of steering)
    J / L       mast camera pan left / right
    C           re-centre the mast camera
    SPACE       release throttle, coast to a stop
    T           toggle autopilot (gentle scripted lap with ToF emergency stop)
    R           reset to the start line
    Q / Esc     quit

All of the above are proper hold-to-move keys, not one-shot taps.

The dashboard also shows the mast RGB camera, the mast depth camera, and a
live top-down lidar plot built from the 24 baked-in rangefinder sensors (see
lidar.py for the arbitrary-ray-count version).

Ackermann is not solved for you here either: each front knuckle gets its own
target from the steering trapezoid, same as demo_car.py.
"""
import argparse
import math
import time

import cv2
import mujoco
import pathlib
MODELS = pathlib.Path(__file__).resolve().parent.parent / "models"
import mujoco.viewer
import numpy as np

from demo_car import ackermann, WB, TRACK, R, STEER_LIM

MAX_SPEED = 1.4        # m/s
SPEED_ACCEL = 1.4       # m/s^2 while W/S held
STEER_ACCEL = 1.4       # rad/s while A/D held
SPEED_RATE = 2.2        # m/s^2, how fast the mechanism chases the speed target
STEER_RATE = 3.5        # rad/s, how fast the mechanism chases the steer target

CAM_PAN_RANGE = (-1.5708, 1.5708)
CAM_TILT_RANGE = (-0.7854, 0.5236)
CAM_SLEW = 1.2           # rad/s while I/J/K/L held

LIDAR_CUTOFF = 8.0
N_LIDAR = 24

DASH_HZ = 12               # dashboard refresh rate (wall clock)
DASH_W, DASH_H = 340, 220  # per-panel render size - big enough to read as a
                            # real control surface, not a thumbnail strip
MAX_SUBSTEPS = 500         # physics steps to catch up per frame, capped so a
                            # slow frame can't spiral into an infinite catch-up
HOLD_WINDOW = 0.25         # a key reads as "still held" this long after the
                            # last repeat event OpenCV's window reports for it


class Keys:
    """Polls the OpenCV dashboard window for key state, including OS key-repeat.

    This is deliberately the ONLY input path. MuJoCo's viewer.key_callback
    (a) only ever fires once per press with no repeat/release, so it can't
    drive a "hold W to accelerate" scheme, and (b) the 3D viewer window has
    hard-wired native shortcuts on most letters (W=wireframe, etc.) that fire
    regardless of any callback, so it can never be repurposed for driving.
    OpenCV's window forwards the OS's native key-repeat stream instead, so
    polling it every loop tick and treating "seen in the last HOLD_WINDOW
    seconds" as held gives real hold-to-move behaviour.
    """

    def __init__(self):
        self.last_seen = {}
        self._prev_edge = {}

    def poll(self):
        now = time.time()
        while True:
            k = cv2.pollKey()
            if k == -1:
                break
            k &= 0xFF
            if 32 <= k < 127:
                self.last_seen[chr(k).lower()] = now
            elif k == 27:
                self.last_seen["esc"] = now

    def held(self, ch, now):
        t = self.last_seen.get(ch)
        return t is not None and (now - t) < HOLD_WINDOW

    def pressed(self, ch, now):
        """Edge-triggered: true only on the tap-down, not while held."""
        h = self.held(ch, now)
        fired = h and not self._prev_edge.get(ch, False)
        self._prev_edge[ch] = h
        return fired


def autopilot_target(t, tof_centre):
    """Same gentle weave + ToF emergency stop as demo_car.py."""
    delta = 0.10 * math.sin(2 * math.pi * t / 4.0)
    speed = 0.0 if 0.0 <= tof_centre < 0.30 else 0.5
    return speed, delta


def lidar_panel(ranges, size=360, cutoff=LIDAR_CUTOFF):
    """Top-down plot of the 24-ray sweep. Forward (body +x) is drawn as up."""
    img = np.full((size, size, 3), 18, np.uint8)
    cx, cy = size // 2, size // 2
    scale = (size / 2 - 14) / cutoff
    for rr in (2, 4, 6, 8):
        if rr > cutoff:
            continue
        cv2.circle(img, (cx, cy), int(rr * scale), (46, 50, 56), 1, cv2.LINE_AA)
        cv2.putText(img, f"{rr:.0f}m", (cx + 3, cy - int(rr * scale) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 96, 104), 1, cv2.LINE_AA)
    cv2.line(img, (cx, 10), (cx, size - 10), (30, 33, 38), 1, cv2.LINE_AA)
    cv2.line(img, (10, cy), (size - 10, cy), (30, 33, 38), 1, cv2.LINE_AA)
    for i, rng in enumerate(ranges):
        if rng < 0 or rng > cutoff:
            continue
        th = 2 * math.pi * i / N_LIDAR
        x, y = rng * math.cos(th), rng * math.sin(th)
        px, py = int(cx - y * scale), int(cy - x * scale)
        cv2.circle(img, (px, py), 3, (72, 224, 240), -1, cv2.LINE_AA)
    tri = np.array([[cx, cy - 9], [cx - 6, cy + 7], [cx + 6, cy + 7]])
    cv2.fillPoly(img, [tri], (90, 150, 255))
    cv2.putText(img, "LIDAR (24 ray)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (200, 205, 210), 1, cv2.LINE_AA)
    return img


def depth_panel(depth, near=0.05, far=4.0):
    d = np.clip(depth, near, far)
    d = 1.0 - (d - near) / (far - near)          # near = bright
    d8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 20), (15, 15, 18), -1)
    cv2.putText(img, text, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (210, 214, 218), 1, cv2.LINE_AA)
    return img


def _apply_control(model, data, A, actual_speed, actual_steer, cam_pan, cam_tilt):
    left, right, _ = ackermann(actual_steer)
    data.ctrl[A["steer_left"]] = left
    data.ctrl[A["steer_right"]] = right
    data.ctrl[A["rear_left_drive"]] = actual_speed / R
    data.ctrl[A["rear_right_drive"]] = actual_speed / R
    data.ctrl[A["camera_pan"]] = cam_pan     # manual only - no auto-follow of steering
    data.ctrl[A["camera_tilt"]] = cam_tilt


def main(autopilot):
    model = mujoco.MjModel.from_xml_path(str(MODELS / "scene_car.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)
    A = {n: aid(n) for n in ("steer_left", "steer_right", "rear_left_drive",
                              "rear_right_drive", "camera_pan", "camera_tilt")}
    tof_c = model.sensor_adr[sid("tof_centre")]
    lidar_adr = [model.sensor_adr[sid(f"lidar{i:02d}")] for i in range(N_LIDAR)]

    dt = model.opt.timestep
    target_speed = target_steer = 0.0
    actual_speed = actual_steer = 0.0
    cam_pan = cam_tilt = 0.0
    autopilot_on = autopilot
    quit_flag = False

    cam_opt = mujoco.MjvOption()
    cam_opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False  # clean camera feeds

    # one renderer, reused for both color and depth -> half the GL overhead
    # of running two full offscreen contexts side by side
    renderer = mujoco.Renderer(model, height=DASH_H, width=DASH_W)

    print(__doc__)
    cv2.namedWindow("AURIGA dashboard", cv2.WINDOW_AUTOSIZE)
    banner = np.zeros((DASH_H, DASH_W * 3, 3), np.uint8)
    cv2.putText(banner, "CLICK THIS WINDOW, THEN USE W/A/S/D TO DRIVE",
                (30, DASH_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (90, 150, 255), 2, cv2.LINE_AA)
    cv2.putText(banner, "(the 3D viewer window will not respond to drive keys)",
                (30, DASH_H // 2 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (150, 155, 160), 1, cv2.LINE_AA)
    cv2.imshow("AURIGA dashboard", banner)
    cv2.waitKey(1)

    keys = Keys()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False  # no lidar lines in the 3D view

        start = time.time()
        last_dash = 0.0
        while viewer.is_running() and not quit_flag:
            keys.poll()
            now = time.time()

            if keys.pressed("t", now):
                autopilot_on = not autopilot_on
                print(f"[autopilot {'ON' if autopilot_on else 'off'}]")
            if keys.pressed("r", now):
                mujoco.mj_resetDataKeyframe(model, data, 0)
                target_speed = target_steer = 0.0
                actual_speed = actual_steer = 0.0
                cam_pan = cam_tilt = 0.0
                start = time.time() - data.time
            if keys.held("q", now) or keys.held("esc", now):
                quit_flag = True
            if keys.held(" ", now):
                target_speed = 0.0
            if keys.pressed("c", now):
                cam_pan = cam_tilt = 0.0

            w, s = keys.held("w", now), keys.held("s", now)
            a, d = keys.held("a", now), keys.held("d", now)
            man_i, man_k = keys.held("i", now), keys.held("k", now)
            man_j, man_l = keys.held("j", now), keys.held("l", now)
            if w or s or a or d:
                autopilot_on = False

            # fixed-timestep catch-up: run as many 1ms physics steps as wall
            # clock demands, capped by MAX_SUBSTEPS. Rendering (below) only
            # happens once per *outer* iteration, so a slow render can no
            # longer stall physics down to a crawl.
            target_t = time.time() - start
            n = 0
            while data.time < target_t and n < MAX_SUBSTEPS:
                if autopilot_on:
                    target_speed, target_steer = autopilot_target(
                        data.time, data.sensordata[tof_c])
                else:
                    if w:
                        target_speed = min(MAX_SPEED, target_speed + SPEED_ACCEL * dt)
                    if s:
                        target_speed = max(-MAX_SPEED / 2, target_speed - SPEED_ACCEL * dt)
                    if a:
                        target_steer = max(-STEER_LIM, target_steer - STEER_ACCEL * dt)
                    if d:
                        target_steer = min(STEER_LIM, target_steer + STEER_ACCEL * dt)

                if man_j:
                    cam_pan = max(CAM_PAN_RANGE[0], cam_pan - CAM_SLEW * dt)
                if man_l:
                    cam_pan = min(CAM_PAN_RANGE[1], cam_pan + CAM_SLEW * dt)
                if man_i:
                    cam_tilt = min(CAM_TILT_RANGE[1], cam_tilt + CAM_SLEW * dt)
                if man_k:
                    cam_tilt = max(CAM_TILT_RANGE[0], cam_tilt - CAM_SLEW * dt)

                actual_speed += np.clip(target_speed - actual_speed,
                                         -SPEED_RATE * dt, SPEED_RATE * dt)
                actual_steer += np.clip(target_steer - actual_steer,
                                         -STEER_RATE * dt, STEER_RATE * dt)
                _apply_control(model, data, A, actual_speed, actual_steer, cam_pan, cam_tilt)
                mujoco.mj_step(model, data)
                n += 1

            viewer.sync()

            if now - last_dash > 1 / DASH_HZ:
                last_dash = now
                renderer.update_scene(data, camera="mast_color", scene_option=cam_opt)
                rgb_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                rgb_bgr = label(rgb_bgr, "MAST COLOR")

                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera="mast_depth", scene_option=cam_opt)
                depth_img = label(depth_panel(renderer.render()), "MAST DEPTH")
                renderer.disable_depth_rendering()

                ranges = [data.sensordata[a] for a in lidar_adr]
                lidar_img = lidar_panel(ranges, size=DASH_H)
                lidar_img = cv2.copyMakeBorder(lidar_img, 0, 0, 0, DASH_W - DASH_H,
                                                cv2.BORDER_CONSTANT, value=(18, 18, 18))
                mode = "AUTOPILOT" if autopilot_on else "MANUAL"
                info = [mode, f"v={actual_speed:+.2f} m/s",
                        f"steer={math.degrees(actual_steer):+.1f} deg",
                        f"cam pan/tilt={math.degrees(cam_pan):+.0f}/{math.degrees(cam_tilt):+.0f} deg",
                        f"t={data.time:6.1f}s"]
                for i, line in enumerate(info):
                    cv2.putText(lidar_img, line, (DASH_H + 15, 45 + 22 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 214, 218), 1, cv2.LINE_AA)

                mosaic = np.hstack([rgb_bgr, depth_img, lidar_img])
                cv2.imshow("AURIGA dashboard", mosaic)

            if n == 0:
                time.sleep(0.001)   # nothing to do; avoid a hot spin loop

    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--autopilot", action="store_true",
                     help="start in scripted-lap mode instead of manual control")
    main(ap.parse_args().autopilot)
