#!/usr/bin/env python3
"""Iterative-MPC path tracking for the AURIGA car, on the figure-8 track.

    python mpc_car.py

Port of mpc_python/mpc_python/mpc_demo_mujoco.py (same cvxpy_mpc package,
same dual-thread physics/solver structure, same on-screen overlays) onto our
own car and track instead of the mushr model:

  * the reference PATH is the actual centerline traced out of the road
    artwork (13Artboard 1@3x.png -> track_path.csv, 80 waypoints extracted
    by thresholding + skeletonizing the drivable-road mask), not hand-picked
    waypoints - the car follows the drawn road, not an abstract line.
  * the MPC's [accel, steer_angle] output is translated into our car's real
    actuators: steer_left/steer_right get the Ackermann-trapezoid knuckle
    angles for that steer_angle (ZOH, same as the position-controlled
    steering PID would track a held target), rear_left/right_drive get the
    accel integrated into a wheel angular velocity (FOH, see demo_car.py's
    docstring note on the same trick).
  * 2 static + 2 moving obstacles sit on the road as mocap bodies
    (scene_mpc.xml); mpc_car.py drives their position every physics step
    from cvxpy_mpc.utils.update_path_obstacles(), the same path-relative
    [distance, speed, lateral_offset] model the reference repo uses, and
    only the closest one inside a fake camera FOV is ever fed to the
    optimiser (its half-plane obstacle constraint only takes one at a time).
"""
from __future__ import annotations

import pathlib
import signal
import sys
import threading
import time

import numpy as np
import numpy.typing as npt

import mujoco
import mujoco.viewer

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "mpc_python" / "mpc_python"))
sys.path.insert(0, str(ROOT / "sim"))
from cvxpy_mpc import MPC
from cvxpy_mpc.utils import (
    compute_path_from_wp,
    compute_errors,
    detect_obstacle_camera,
    ego_to_global,
    get_ref_trajectory,
    update_path_obstacles,
)

from demo_car import ackermann, R

HERE = pathlib.Path(__file__).resolve().parent
MODELS = ROOT / "models"
N_LAPS = 3
TARGET_SPEED = 0.6
SENSOR_MAX_RANGE = 3.0
SENSOR_FOV_DEG = 100.0

# Hard emergency-brake backstop (see main()), independent of what the MPC
# commands. The trigger distance is SPEED-DEPENDENT (physical stopping
# distance + a fixed pad) rather than one fixed number: a fixed threshold
# either has to be loose enough not to slam the brakes mid-maneuver during
# a normal, already-slow, controlled pass close by an obstacle (in which
# case it's too loose to actually stop a fast approach in time), or tight
# enough to stop a fast approach (in which case it fires early, right in the
# middle of a legitimate pass).
#
# Braking all the way to a dead stop turned out to be its own trap: this is
# a KINEMATIC bicycle model, so a stationary car doesn't turn at all no
# matter what the steering angle is - only rolling forward with the wheels
# turned changes heading. Once fully stopped nose-to-obstacle, any attempt
# to creep forward immediately re-triggers the brake at the same spot
# (near-zero speed -> near-zero required stopping distance -> "still too
# close" forever), so the car can never work its way into the maneuver the
# MPC is actually trying to execute. Flooring the brake at a slow creep
# instead keeps steering live, so the car can still work its way around.
EMERGENCY_DECEL = 3.0     # m/s^2 - stronger than mpc.max_acc, not rate-limited
EMERGENCY_PAD = 0.03      # m, extra margin added to the computed stopping distance
CREEP_SPEED = 0.10        # m/s - brake floor, not a full stop (see above)

# Road is ~1.0-1.09 m wide (measured off the artwork by extract_path.py) and
# the car is 0.23 m wide, so there's real room to place obstacles that force
# a swerve without blocking the road: radii and offsets sized against that
# ~1 m width, keeping |lateral_offset| + radius comfortably inside the
# +/-0.5 m road half-width. lap_frac is a FRACTION of one lap's length,
# resolved once the path is loaded; speed=0 -> static. lateral_offset is
# metres from the centreline (+left/-right of the direction of travel) -
# never exactly 0.0 (see update_path_obstacles' note on the half-plane
# normal degenerating there).
OBSTACLES = [
    {"lap_frac": 0.18, "speed": 0.0, "radius": 0.16, "lateral_offset": 0.22, "mocap": "obstacle_static_0"},
    {"lap_frac": 0.62, "speed": 0.0, "radius": 0.16, "lateral_offset": -0.20, "mocap": "obstacle_static_1"},
    {"lap_frac": 0.38, "speed": 0.08, "radius": 0.14, "lateral_offset": -0.18, "mocap": "obstacle_moving_0"},
    {"lap_frac": 0.85, "speed": -0.07, "radius": 0.14, "lateral_offset": 0.20, "mocap": "obstacle_moving_1"},
]


def load_track_path(n_laps: int = N_LAPS) -> npt.NDArray[np.float64]:
    """track_path.csv (road centerline extracted from the artwork) -> a
    smooth, dense, closed, multi-lap [x,y,heading] reference path."""
    raw = np.loadtxt(HERE / "track_path.csv", delimiter=",", skiprows=1)
    xs, ys = raw[:, 0], raw[:, 1]
    # close the loop explicitly so the spline doesn't leave a seam
    xs = np.append(xs, xs[0])
    ys = np.append(ys, ys[0])
    single_lap = compute_path_from_wp(xs, ys, step=0.05)
    laps = [single_lap] + [single_lap[:, 1:] for _ in range(n_laps - 1)]
    return np.concatenate(laps, axis=1)


class SharedData:
    """Same split as mpc_demo_mujoco.py: MPC solves on its own thread."""

    def __init__(self) -> None:
        self.lock: threading.Lock = threading.Lock()
        self.state: npt.NDArray[np.float64] = np.zeros(4)
        self.goal_reached: bool = False
        self.is_active: bool = True
        self.mpc_accel: float = 0.0
        self.mpc_steer: float = 0.0
        self.x_mpc_world: npt.NDArray[np.float64] | None = None
        self.mpc_elapsed: float = 0.0
        self.obstacle: tuple[float, float, float, float, float] | None = None


def controller_loop(mpc: MPC, path, shared: SharedData, target_speed: float) -> None:
    while True:
        start_time = time.time()
        with shared.lock:
            if not shared.is_active or shared.goal_reached:
                break
            current_state = shared.state.copy()
            global_obs = shared.obstacle
        last_control = (shared.mpc_accel, shared.mpc_steer)
        elapsed = shared.mpc_elapsed

        # NOTE: unlike mpc_demo_mujoco.py's open point-A-to-B path, ours is a
        # closed loop repeated N_LAPS times back to back - path[:, -1] sits
        # at the exact same (x, y) as the start/every lap-return point, so a
        # spatial "am I near the goal" check would fire after lap 1 every
        # time. Lap counting is done by the physics thread via odometry
        # instead (see main()); this thread just stops when that thread
        # flips shared.goal_reached.

        # delay compensation: predict where we'll actually be when this
        # command lands, same reasoning as mpc_demo_mujoco.py
        pred_state = current_state.copy()
        v, theta = pred_state[2], pred_state[3]
        a, delta = last_control
        L = mpc.wheelbase
        pred_state[0] += v * np.cos(theta) * elapsed
        pred_state[1] += v * np.sin(theta) * elapsed
        pred_state[2] += a * elapsed
        pred_state[3] += (v * np.tan(delta) / L) * elapsed

        target = get_ref_trajectory(pred_state, path, target_speed, mpc.control_horizon * mpc.dt, mpc.dt)
        pred_ego_state = [0.0, 0.0, pred_state[2], 0.0]

        if global_obs is not None:
            gx, gy, r, vx, vy = global_obs
            dx, dy = gx - pred_state[0], gy - pred_state[1]
            ct, st = np.cos(-pred_state[3]), np.sin(-pred_state[3])
            pred_obstacle = (dx * ct - dy * st, dy * ct + dx * st, r, vx * ct - vy * st, vy * ct + vx * st)
        else:
            pred_obstacle = None

        x_mpc, u_mpc = mpc.solve(pred_ego_state, target, verbose=False, obstacle=pred_obstacle)
        if x_mpc is None:
            # Solver failed -> cvxpy_mpc's fallback brakes but zeroes the
            # steering command. Fine on an open road; on this closed loop it
            # would snap the car dead straight off whatever curve it's
            # mid-corner on. Keep the last known-good steering angle and
            # only trust the library's deceleration profile.
            control = (u_mpc[0, 0], last_control[1])
            # mpc._previous_command is what solve() anchors the NEXT call's
            # max_d_steer rate-limit constraint to. The library just set its
            # steer row to 0 for the failed call; left alone, the next solve
            # would be constrained to near-0 steer relative to a "last
            # command" that was never actually applied - while the real
            # steer we just held could be far from 0. That mismatch can make
            # the next solve infeasible too, and the one after that, in a
            # cascade that never recovers (observed: it ends in the car
            # parked at a fixed, wrong, off-path position, seemingly forever
            # unable to solve again). Keep the library's own warm-start
            # state honest about what was actually commanded.
            mpc._previous_command[1, :] = last_control[1]
        else:
            control = (u_mpc[0, 0], u_mpc[1, 0])
        elapsed = time.time() - start_time

        with shared.lock:
            shared.mpc_accel, shared.mpc_steer = control
            shared.mpc_elapsed = elapsed
            shared.x_mpc_world = ego_to_global(pred_state, x_mpc) if x_mpc is not None else None

        sleep_time = max(0.0, mpc.dt - (time.time() - start_time))
        time.sleep(sleep_time)


def get_state(data: mujoco.MjData, bid: int) -> npt.NDArray[np.float64]:
    rot = data.xmat[bid].reshape(3, 3)
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    speed = np.linalg.norm(data.qvel[0:2])
    return np.array([data.xpos[bid][0], data.xpos[bid][1], speed, yaw])


def apply_control(model, data, A, mpc_steer: float, speed_mps: float) -> None:
    left, right, _ = ackermann(mpc_steer)
    data.ctrl[A["steer_left"]] = left
    data.ctrl[A["steer_right"]] = right
    data.ctrl[A["rear_left_drive"]] = speed_mps / R
    data.ctrl[A["rear_right_drive"]] = speed_mps / R


# ---- viewer overlays: identical in spirit to mpc_demo_mujoco.py -----------

def draw_path(viewer, path) -> None:
    for i in range(0, path.shape[1] - 1, 3):   # every 3rd segment - it's a long multi-lap path
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        p1 = np.array([path[0, i], path[1, i], 0.03])
        p2 = np.array([path[0, min(i + 3, path.shape[1] - 1)], path[1, min(i + 3, path.shape[1] - 1)], 0.03])
        mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=np.array([0.012, 0, 0]),
                             pos=np.zeros(3), mat=np.eye(3).ravel(), rgba=np.array([0, 0.6, 1, 1], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, p1, p2)
        viewer.user_scn.ngeom += 1


def draw_trail(viewer, x_hist, y_hist, downsample: int = 15) -> None:
    if len(x_hist) < 2:
        return
    for i in range(0, len(x_hist) - 1, downsample):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        alpha = (i + 1) / len(x_hist) * 0.8
        p1 = np.array([x_hist[i], y_hist[i], 0.005])
        p2 = np.array([x_hist[i + 1], y_hist[i + 1], 0.005])
        mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=np.array([0.02, 0, 0]),
                             pos=np.zeros(3), mat=np.eye(3).ravel(), rgba=np.array([1, 0, 0, alpha], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.02, p1, p2)
        viewer.user_scn.ngeom += 1


def draw_mpc_preview(viewer, x_mpc_world) -> None:
    for i in range(x_mpc_world.shape[1]):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.025, 0, 0]),
                             pos=np.array([x_mpc_world[0, i], x_mpc_world[1, i], 0.01]),
                             mat=np.eye(3).ravel(), rgba=np.array([0, 1, 0, 0.6], dtype=np.float32))
        viewer.user_scn.ngeom += 1


def draw_sensor_fov(viewer, x, y, heading, max_range, fov_deg) -> None:
    fov_rad = np.radians(fov_deg)
    right_angle, left_angle = heading - fov_rad / 2.0, heading + fov_rad / 2.0
    origin = np.array([x, y, 0.0])
    rgba = np.array([0.8, 0.8, 0, 0.15], dtype=np.float32)
    for angle in (right_angle, left_angle):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        end = np.array([x + max_range * np.cos(angle), y + max_range * np.sin(angle), 0.0])
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=np.array([0.01, 0, 0]),
                             pos=np.zeros(3), mat=np.eye(3).ravel(), rgba=rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.008, origin, end)
        viewer.user_scn.ngeom += 1
    prev = None
    for i in range(21):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        angle = right_angle + (i / 20) * fov_rad
        p = np.array([x + max_range * np.cos(angle), y + max_range * np.sin(angle), 0.0])
        if prev is not None:
            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=np.array([0.008, 0, 0]),
                                 pos=np.zeros(3), mat=np.eye(3).ravel(), rgba=rgba)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.008, prev, p)
            viewer.user_scn.ngeom += 1
        prev = p


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODELS / "scene_mpc.xml"))
    data = mujoco.MjData(model)

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    A = {n: aid(n) for n in ("steer_left", "steer_right", "rear_left_drive", "rear_right_drive")}
    steer_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "front_left_steer_joint")
    steer_qaddr = model.jnt_qposadr[steer_jnt]

    path = load_track_path()
    cdist = np.zeros(path.shape[1])
    cdist[1:] = np.cumsum(np.hypot(np.diff(path[0]), np.diff(path[1])))
    lap_length = cdist[np.argmin(np.abs(cdist - cdist[-1] / N_LAPS))]  # length of ONE lap

    path_obstacles = []
    mocap_ids = {}
    for obs in OBSTACLES:
        mocap_ids[obs["mocap"]] = model.body_mocapid[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obs["mocap"])
        ]
        path_obstacles.append({
            "distance": obs["lap_frac"] * lap_length,
            "speed": obs["speed"],
            "radius": obs["radius"],
            "lateral_offset": obs["lateral_offset"],
        })
    dynamic_obstacle_list = update_path_obstacles(path_obstacles, path, 0.0)

    def sync_obstacle_mocaps():
        for obs_cfg, (ox, oy, _, _, _) in zip(OBSTACLES, dynamic_obstacle_list):
            data.mocap_pos[mocap_ids[obs_cfg["mocap"]]] = [ox, oy, 0.10]

    # spawn the car at the path start, facing the initial path heading
    start_x, start_y, start_yaw = path[0, 0], path[1, 0], path[2, 0]
    data.qpos[0:3] = [start_x, start_y, 0.048]
    data.qpos[3:7] = [np.cos(start_yaw / 2), 0, 0, np.sin(start_yaw / 2)]
    sync_obstacle_mocaps()
    mujoco.mj_forward(model, data)

    mpc = MPC(str(HERE / "mpc_car.yaml"), horizon_time=3.0)

    shared = SharedData()
    mpc_thread = threading.Thread(
        target=controller_loop, args=(mpc, path, shared, TARGET_SPEED), daemon=True
    )

    shutdown_flag = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: shutdown_flag.set())

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False  # no lidar lines cluttering the view
        viewer.cam.lookat[:] = [start_x, start_y, 0.0]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -35

        fps = 60.0
        render_dt = 1.0 / fps
        x_hist, y_hist, cte_hist, heading_err_hist = [], [], [], []
        cte_rmse = heading_rmse = -1.0

        sim_start_time = time.perf_counter()
        mpc_thread.start()
        speed_accum = 0.0        # FOH: mpc_accel gets integrated into this every physics step
        dist_traveled = 0.0      # odometry-based lap counter (see controller_loop's note on why
                                  # a spatial "near the goal" check can't work on a repeating loop)

        try:
            while viewer.is_running() and not shutdown_flag.is_set():
                if shared.goal_reached:
                    viewer.set_texts([(None, None,
                        f"GOAL REACHED ({N_LAPS} laps)\n"
                        f"Final RMSE:   CTE {cte_rmse:.3f} m  |  heading {heading_rmse:.1f} deg\n", "")])
                    viewer.sync()
                    print("\nGoal reached! Close the viewer window or press CTRL-C to exit.")
                    while viewer.is_running():
                        time.sleep(0.1)
                    break

                elapsed_real_time = time.perf_counter() - sim_start_time

                # State publish and control read used to happen once per
                # OUTER (render) iteration, then get applied to every physics
                # step needed to catch up to elapsed_real_time. If rendering
                # ever lags, that catch-up burst can be hundreds of steps
                # long, all driven by one now-stale accel/steer command and
                # feeding the MPC thread a now-stale position back - on a fast
                # corner near a wall that's exactly the kind of latency that
                # sends the car straight off the road before anyone notices.
                # Refreshing every REFRESH_STEPS physics steps instead keeps
                # control fresh independent of render cadence, at a small,
                # fixed CPU cost (a lock + a few trig calls) rather than a
                # cost proportional to however long rendering just took.
                REFRESH_STEPS = 20   # ~20ms of physics between refreshes
                mpc_accel = mpc_steer = 0.0
                emergency_brake = False
                step_i = 0
                while data.time < elapsed_real_time and not shutdown_flag.is_set():
                    if step_i % REFRESH_STEPS == 0:
                        step_state = get_state(data, bid)
                        step_obs = detect_obstacle_camera(
                            dynamic_obstacle_list, step_state[0], step_state[1], step_state[3],
                            SENSOR_MAX_RANGE, SENSOR_FOV_DEG,
                        )
                        with shared.lock:
                            shared.state[:] = step_state
                            shared.obstacle = step_obs
                            mpc_accel, mpc_steer = shared.mpc_accel, shared.mpc_steer

                        # Hard AEB-style backstop, independent of the MPC.
                        # Obstacle avoidance in cvxpy_mpc is a SOFT (slack-
                        # penalized) constraint: it's minimizing a weighted
                        # sum of path-tracking + avoidance + control-effort
                        # cost, not enforcing "never get close" as a rule. If
                        # a full stop scores worse on that sum than clipping
                        # the obstacle for one step (or detection simply
                        # came in too late/narrow an FOV to react), it WILL
                        # clip it - there's nothing else standing between it
                        # and a collision. This mirrors demo_car.py's ToF
                        # emergency stop, but keyed on obstacle clearance.
                        emergency_brake = False
                        if step_obs is not None:
                            ox, oy, orad, _, _ = step_obs
                            clearance = (np.hypot(step_state[0] - ox, step_state[1] - oy)
                                         - orad - mpc.width / 2)
                            stopping_dist = speed_accum ** 2 / (2 * EMERGENCY_DECEL) + EMERGENCY_PAD
                            emergency_brake = clearance < stopping_dist
                    step_i += 1

                    if emergency_brake:
                        speed_accum = max(CREEP_SPEED, speed_accum - EMERGENCY_DECEL * model.opt.timestep)
                    else:
                        speed_accum = np.clip(speed_accum + mpc_accel * model.opt.timestep, 0.0, mpc.max_speed)
                    dist_traveled += speed_accum * model.opt.timestep
                    apply_control(model, data, A, mpc_steer, speed_accum)
                    dynamic_obstacle_list[:] = update_path_obstacles(path_obstacles, path, model.opt.timestep)
                    sync_obstacle_mocaps()
                    mujoco.mj_step(model, data)

                if dist_traveled >= N_LAPS * lap_length:
                    with shared.lock:
                        shared.goal_reached = True

                # current_state/detected_obs for the HUD & overlays below
                # reflect where physics actually ended up this frame
                current_state = get_state(data, bid)
                detected_obs = detect_obstacle_camera(
                    dynamic_obstacle_list, current_state[0], current_state[1], current_state[3],
                    SENSOR_MAX_RANGE, SENSOR_FOV_DEG,
                )
                with shared.lock:
                    mpc_elapsed = shared.mpc_elapsed
                    x_mpc_world = shared.x_mpc_world

                x_hist.append(current_state[0])
                y_hist.append(current_state[1])
                cte, heading_err = compute_errors(current_state, path)
                cte_hist.append(cte)
                heading_err_hist.append(np.degrees(heading_err))
                cte_rmse = np.sqrt(np.mean(np.square(cte_hist)))
                heading_rmse = np.sqrt(np.mean(np.square(heading_err_hist)))

                viewer.cam.lookat[:] = [current_state[0], current_state[1], 0.0]

                # obstacles render via their own mocap-body geoms, nothing to draw for them here
                viewer.user_scn.ngeom = 0
                draw_path(viewer, path)
                draw_sensor_fov(viewer, current_state[0], current_state[1], current_state[3],
                                 SENSOR_MAX_RANGE, SENSOR_FOV_DEG)
                draw_trail(viewer, x_hist, y_hist)
                if x_mpc_world is not None:
                    draw_mpc_preview(viewer, x_mpc_world)

                actual_steer = np.degrees(data.qpos[steer_qaddr])
                lap_progress = dist_traveled / lap_length
                viewer.set_texts([(None, None,
                    f"AURIGA MPC Demo\n"
                    f"state:  v {current_state[2]:.2f} m/s  |  steer {actual_steer:.1f} deg\n"
                    f"MPC:    accel {mpc_accel:.2f} m/s2  |  steer {np.degrees(mpc_steer):.1f} deg  |  {mpc_elapsed*1000:.0f} ms\n"
                    f"error:  CTE {cte:.3f} m  |  heading {np.degrees(heading_err):.1f} deg\n"
                    f"RMSE:   CTE {cte_rmse:.3f} m  |  heading {heading_rmse:.1f} deg\n"
                    f"lap:    {lap_progress:.2f} / {N_LAPS}\n"
                    f"avoid:  {'YES' if detected_obs is not None else 'no'}\n", "")])
                viewer.sync()

                time_until_next = render_dt - (time.perf_counter() - elapsed_real_time - sim_start_time)
                if time_until_next > 0:
                    time.sleep(time_until_next)
        except Exception as e:
            print(e)
        finally:
            shared.is_active = False
            mpc_thread.join(timeout=1.0)
            if mpc_thread.is_alive():
                print("MPC thread is still alive...")
            viewer.clear_texts()


if __name__ == "__main__":
    main()
