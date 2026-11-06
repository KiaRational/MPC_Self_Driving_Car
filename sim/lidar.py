#!/usr/bin/env python3
"""Lidar for the AURIGA models.

Two ways to get a scan:

  * the 24 `rangefinder` sensors baked into each model, read straight out of
    `data.sensordata` every step - cheap, fixed, and available to any controller;
  * `scan2d()` / `scan3d()` below, which cast arbitrarily many rays with
    `mj_multiRay` on demand - use these when you want a real 360x1 sweep or a
    multi-beam puck without putting 900 sites in the XML.

    python lidar.py                 # both robots, prints stats + writes plots
    python lidar.py manipulator     # one of them
"""
import sys

import mujoco
import pathlib
MODELS = pathlib.Path(__file__).resolve().parent.parent / "models"
import numpy as np

RANGE_MAX = {"manipulator": 12.0, "car": 8.0}
SCENES = {"manipulator": "scene_manipulator.xml", "car": "scene_car.xml"}


def _cast(model, data, origin, dirs, exclude_body, cutoff, self_hits=True):
    n = len(dirs)
    dist = np.zeros(n)
    geomid = np.zeros(n, dtype=np.int32)
    # the robots keep their geoms in groups 2 (visual) and 3 (collision), so
    # masking those to 0 turns the scan into a world-only sweep
    mask = None if self_hits else np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
    mujoco.mj_multiRay(
        m=model, d=data,
        pnt=np.asarray(origin, dtype=np.float64),
        vec=np.asarray(dirs, dtype=np.float64).flatten(),
        geomgroup=mask, flg_static=1, bodyexclude=exclude_body,
        geomid=geomid, dist=dist, normal=None, nray=n, cutoff=cutoff)
    dist[(dist < 0) | (dist > cutoff)] = np.inf     # -1 means the ray hit nothing
    return dist, geomid


def _frame(model, data, body="lidar_link"):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    return bid, data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def scan2d(model, data, n_rays=360, cutoff=12.0, body="lidar_link",
           self_hits=True):
    """One horizontal sweep. Returns (bearings, ranges) in the lidar frame.

    self_hits=False ignores the robot's own geoms, which is the usual thing to
    want; the baked-in rangefinder sensors always see the robot.
    """
    bid, origin, R = _frame(model, data, body)
    th = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    local = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], axis=1)
    dist, _ = _cast(model, data, origin, local @ R.T, bid, cutoff, self_hits)
    return th, dist


def scan3d(model, data, n_rays=360, elevations=(-15, -5, 5, 15),
           cutoff=12.0, body="lidar_link"):
    """Multi-beam puck. Returns an (n_beams, n_rays) range image."""
    bid, origin, R = _frame(model, data, body)
    th = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    out = np.zeros((len(elevations), n_rays))
    for i, el in enumerate(np.radians(elevations)):
        local = np.stack([np.cos(el) * np.cos(th),
                          np.cos(el) * np.sin(th),
                          np.full_like(th, np.sin(el))], axis=1)
        out[i], _ = _cast(model, data, origin, local @ R.T, bid, cutoff)
    return th, out


def to_points(bearings, ranges):
    """Polar scan -> Nx3 points in the lidar frame, dropping the misses."""
    ok = np.isfinite(ranges)
    b, r = bearings[ok], ranges[ok]
    return np.stack([r * np.cos(b), r * np.sin(b), np.zeros_like(r)], axis=1)


def _plot(kind, bearings, ranges, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5, 5), facecolor="#14171c")
    ax = fig.add_subplot(projection="polar", facecolor="#1b1f26")
    hit = np.isfinite(ranges)
    ax.scatter(bearings[hit], ranges[hit], s=6, c="#46c9d4")
    ax.set_title(f"{kind}: 2D scan (lidar frame)", color="#e7ecf2", fontsize=10)
    ax.tick_params(colors="#8a96a5", labelsize=7)
    ax.spines["polar"].set_color("#2b333d")
    ax.grid(color="#2b333d")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"  wrote {path}")


def run(kind):
    model = mujoco.MjModel.from_xml_path(str(MODELS / SCENES[kind]))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(int(0.8 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    cutoff = RANGE_MAX[kind]
    bearings, ranges = scan2d(model, data, n_rays=360, cutoff=cutoff)
    hits = np.isfinite(ranges)
    print(f"{kind}: 360-ray sweep -> {hits.sum()} returns, "
          f"nearest {ranges[hits].min():.3f} m, farthest {ranges[hits].max():.3f} m")
    _, world = scan2d(model, data, n_rays=360, cutoff=cutoff, self_hits=False)
    w = np.isfinite(world)
    print(f"  ignoring the robot's own geoms: {w.sum()} returns, "
          f"nearest {world[w].min():.3f} m")

    _, cube = scan3d(model, data, n_rays=180, cutoff=cutoff)
    print(f"  4-beam 3D scan: {np.isfinite(cube).sum()} of {cube.size} rays return")

    # the baked-in sensors should agree with the on-demand cast
    adr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "lidar00")
    if adr >= 0:
        baked = np.array([data.sensordata[model.sensor_adr[adr + i]] for i in range(24)])
        cast = scan2d(model, data, n_rays=24, cutoff=cutoff)[1]
        both = np.isfinite(cast) & (baked > 0)
        if both.any():
            err = np.abs(baked[both] - cast[both]).max()
            print(f"  baked sensors vs mj_multiRay: max disagreement {err:.4f} m")
    _plot(kind, bearings, ranges, f"lidar_{kind}.png")


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["manipulator", "car"]):
        run(k)
