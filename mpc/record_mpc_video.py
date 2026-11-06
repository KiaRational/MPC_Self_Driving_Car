#!/usr/bin/env python3
"""Headless recorder for the AURIGA MPC + obstacle-avoidance demo.

    MUJOCO_GL=egl    python mpc/record_mpc_video.py            # GPU
    MUJOCO_GL=osmesa python mpc/record_mpc_video.py --laps 1   # CPU-only

Same controller as mpc_car.py (same cvxpy_mpc MPC, same obstacle model,
same AEB backstop, same overlays) but stepped deterministically in *sim*
time instead of wall-clock time: the MPC is re-solved every mpc.dt of
simulated time and its command applied immediately, so the result does not
depend on how fast the machine renders. Writes an MP4 plus a README-sized GIF.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

import mpc_car as mc
from mpc_car import MPC, compute_errors, detect_obstacle_camera, ego_to_global, \
    get_ref_trajectory, update_path_obstacles

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ROOT / "models"


class _Shim:
    """mpc_car's draw_* helpers expect a viewer with .user_scn."""
    def __init__(self, scn):
        self.user_scn = scn


def solve_mpc(mpc, path, state, obs, last_control):
    target = get_ref_trajectory(state, path, mc.TARGET_SPEED, mpc.control_horizon * mpc.dt, mpc.dt)
    pred_obstacle = None
    if obs is not None:
        gx, gy, r, vx, vy = obs
        dx, dy = gx - state[0], gy - state[1]
        ct, st = np.cos(-state[3]), np.sin(-state[3])
        pred_obstacle = (dx * ct - dy * st, dy * ct + dx * st, r, vx * ct - vy * st, vy * ct + vx * st)
    x_mpc, u_mpc = mpc.solve([0.0, 0.0, state[2], 0.0], target, verbose=False, obstacle=pred_obstacle)
    if x_mpc is None:  # same fallback as mpc_car.controller_loop
        mpc._previous_command[1, :] = last_control[1]
        return (u_mpc[0, 0], last_control[1]), None
    return (u_mpc[0, 0], u_mpc[1, 0]), ego_to_global(state, x_mpc)


def hud(img: np.ndarray, lines: list[str], font, title_font, avoiding: bool) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    w = 470
    h = 46 + 28 * len(lines)
    d.rounded_rectangle([18, 18, 18 + w, 18 + h], radius=14, fill=(10, 14, 24, 175))
    d.text((34, 26), "AURIGA  ·  MPC + obstacle avoidance", font=title_font, fill=(255, 255, 255, 255))
    for i, ln in enumerate(lines):
        d.text((34, 62 + 28 * i), ln, font=font, fill=(215, 225, 240, 255))
    if avoiding:
        d.rounded_rectangle([18, 30 + h, 230, 72 + h], radius=10, fill=(230, 70, 60, 220))
        d.text((34, 38 + h), "OBSTACLE  AVOID", font=font, fill=(255, 255, 255, 255))
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default=str(ROOT / "media" / "auriga_mpc_obstacle_avoidance"))
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODELS / "scene_mpc.xml"))
    data = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    A = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
         for n in ("steer_left", "steer_right", "rear_left_drive", "rear_right_drive")}
    steer_qaddr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "front_left_steer_joint")]

    path = mc.load_track_path()
    cdist = np.zeros(path.shape[1])
    cdist[1:] = np.cumsum(np.hypot(np.diff(path[0]), np.diff(path[1])))
    lap_length = cdist[np.argmin(np.abs(cdist - cdist[-1] / mc.N_LAPS))]

    mocap_ids, path_obstacles = {}, []
    for obs in mc.OBSTACLES:
        mocap_ids[obs["mocap"]] = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obs["mocap"])]
        path_obstacles.append({"distance": obs["lap_frac"] * lap_length, "speed": obs["speed"],
                               "radius": obs["radius"], "lateral_offset": obs["lateral_offset"]})
    dyn_obs = update_path_obstacles(path_obstacles, path, 0.0)

    def sync_mocaps():
        for cfg, (ox, oy, _, _, _) in zip(mc.OBSTACLES, dyn_obs):
            data.mocap_pos[mocap_ids[cfg["mocap"]]] = [ox, oy, 0.10]

    sx, sy, syaw = path[0, 0], path[1, 0], path[2, 0]
    data.qpos[0:3] = [sx, sy, 0.048]
    data.qpos[3:7] = [np.cos(syaw / 2), 0, 0, np.sin(syaw / 2)]
    sync_mocaps()
    mujoco.mj_forward(model, data)

    mpc = MPC(str(HERE / "mpc_car.yaml"), horizon_time=3.0)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width, max_geom=20000)
    mini_w, mini_h = args.width * 3 // 8, args.height * 3 // 8
    mini = mujoco.Renderer(model, height=mini_h, width=mini_w, max_geom=20000)
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.0, -90, -35
    top = mujoco.MjvCamera()
    top.lookat[:] = [0, 0, 0]
    top.distance, top.azimuth, top.elevation = 13.0, 90, -89.9

    try:
        font = ImageFont.load_default(size=20)
        title_font = ImageFont.load_default(size=23)
    except TypeError:
        font = title_font = ImageFont.load_default()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    writer = imageio_ffmpeg.write_frames(str(out.with_suffix(".mp4")), (args.width, args.height),
                                         fps=args.fps, codec="libx264", quality=8,
                                         output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    writer.send(None)

    dt = model.opt.timestep
    steps_per_frame = max(1, round(1.0 / (args.fps * dt)))
    steps_per_mpc = max(1, round(mpc.dt / dt))
    REFRESH = 20
    control = (0.0, 0.0)
    x_mpc_world = None
    speed = dist = 0.0
    brake = False
    xh, yh, cte_h, he_h = [], [], [], []
    n_avoid_frames = n_brake = 0
    min_clear = np.inf
    step = 0
    goal = args.laps * lap_length

    while dist < goal:
        state = mc.get_state(data, bid)
        obs_now = detect_obstacle_camera(dyn_obs, state[0], state[1], state[3], mc.SENSOR_MAX_RANGE, mc.SENSOR_FOV_DEG)
        if step % steps_per_mpc == 0:
            control, xw = solve_mpc(mpc, path, state, obs_now, control)
            x_mpc_world = xw
        if step % REFRESH == 0:
            brake = False
            if obs_now is not None:
                ox, oy, orad, _, _ = obs_now
                clear = np.hypot(state[0] - ox, state[1] - oy) - orad - mpc.width / 2
                brake = clear < speed ** 2 / (2 * mc.EMERGENCY_DECEL) + mc.EMERGENCY_PAD
                n_brake += brake
        for o in dyn_obs:
            min_clear = min(min_clear, np.hypot(state[0] - o[0], state[1] - o[1]) - o[2] - mpc.width / 2)
        if brake:
            speed = max(mc.CREEP_SPEED, speed - mc.EMERGENCY_DECEL * dt)
        else:
            speed = float(np.clip(speed + control[0] * dt, 0.0, mpc.max_speed))
        dist += speed * dt
        mc.apply_control(model, data, A, control[1], speed)
        dyn_obs[:] = update_path_obstacles(path_obstacles, path, dt)
        sync_mocaps()
        mujoco.mj_step(model, data)
        step += 1

        if step % steps_per_frame:
            continue
        cs = mc.get_state(data, bid)
        det = detect_obstacle_camera(dyn_obs, cs[0], cs[1], cs[3], mc.SENSOR_MAX_RANGE, mc.SENSOR_FOV_DEG)
        n_avoid_frames += det is not None
        xh.append(cs[0]); yh.append(cs[1])
        cte, he = compute_errors(cs, path)
        cte_h.append(cte); he_h.append(np.degrees(he))

        cam.lookat[:] = [cs[0], cs[1], 0.0]
        renderer.update_scene(data, camera=cam, scene_option=opt)
        v = _Shim(renderer.scene)
        mc.draw_path(v, path)
        mc.draw_sensor_fov(v, cs[0], cs[1], cs[3], mc.SENSOR_MAX_RANGE, mc.SENSOR_FOV_DEG)
        mc.draw_trail(v, xh, yh, downsample=2)
        if x_mpc_world is not None:
            mc.draw_mpc_preview(v, x_mpc_world)
        img = renderer.render().copy()

        mini.update_scene(data, camera=top, scene_option=opt)
        mv = _Shim(mini.scene)
        mc.draw_trail(mv, xh, yh, downsample=2)
        m = mini.render()
        pad = 18
        y0, x0 = args.height - mini_h - pad, args.width - mini_w - pad
        img[y0 - 3:y0 + mini_h + 3, x0 - 3:x0 + mini_w + 3] = 255
        img[y0:y0 + mini_h, x0:x0 + mini_w] = m

        lines = [
            f"speed   {cs[2]:.2f} m/s     steer {np.degrees(data.qpos[steer_qaddr]):+5.1f} deg",
            f"MPC     accel {control[0]:+.2f}     steer {np.degrees(control[1]):+5.1f} deg",
            f"error   CTE {cte:+.3f} m   heading {np.degrees(he):+5.1f} deg",
            f"RMSE    CTE {np.sqrt(np.mean(np.square(cte_h))):.3f} m",
            f"lap     {dist / lap_length:.2f}   t = {data.time:5.1f} s",
        ]
        writer.send(np.ascontiguousarray(hud(img, lines, font, title_font, det is not None)))
        if len(xh) % 150 == 0:
            print(f"t={data.time:6.1f}s  lap={dist / lap_length:.2f}  frames={len(xh)}", flush=True)

    for _ in range(args.fps):  # hold the last frame for a second
        writer.send(np.ascontiguousarray(hud(img, lines, font, title_font, False)))
    writer.close()

    mp4 = out.with_suffix(".mp4")
    gif = out.with_suffix(".gif")
    subprocess.run([ffmpeg, "-y", "-i", str(mp4), "-vf",
                    "setpts=PTS/1.6,fps=10,scale=640:-1:flags=lanczos,split[a][b];"
                    "[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                    str(gif)], check=True, capture_output=True)
    print(f"\nwrote {mp4} ({mp4.stat().st_size / 1e6:.1f} MB) and {gif} ({gif.stat().st_size / 1e6:.1f} MB)")
    print(f"sim time {data.time:.1f} s, laps {dist / lap_length:.2f}, "
          f"CTE RMSE {np.sqrt(np.mean(np.square(cte_h))):.3f} m, heading RMSE {np.sqrt(np.mean(np.square(he_h))):.1f} deg, "
          f"min obstacle clearance {min_clear:+.3f} m, avoid frames {n_avoid_frames}, AEB triggers {n_brake}")


if __name__ == "__main__":
    main()
