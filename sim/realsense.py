#!/usr/bin/env python3
"""RGB-D from the AURIGA D435 models.

MuJoCo has no "depth sensor" element - a `<camera>` is just a viewpoint, and
depth comes from the renderer. So the RealSense is modelled as two cameras
matching the real module's two streams (colour 69x42 deg, depth 87x58 deg,
offset by the 17.5 mm baseline), and you pull depth by asking the renderer for
it:

    renderer.enable_depth_rendering()
    depth = renderer.render()          # metres, along the camera z axis

    MUJOCO_GL=osmesa python realsense.py            # both robots
    MUJOCO_GL=osmesa python realsense.py car
"""
import os
import sys

import mujoco
import pathlib
MODELS = pathlib.Path(__file__).resolve().parent.parent / "models"
import numpy as np
from PIL import Image

RIGS = {
    "manipulator": dict(scene="scene_manipulator.xml",
                        color="wrist_color", depth="wrist_depth",
                        ctrl={"shoulder_pitch": -0.55, "elbow": 1.05,
                              "wrist_pitch": 0.75, "gripper": 0.030},
                        clip=3.0),
    "car": dict(scene="scene_car.xml", color="mast_color", depth="mast_depth",
                ctrl={"camera_tilt": -0.10}, clip=6.0),
}
W, H = 848, 480


def scene_opt():
    """Rangefinder rays are drawn by default; 27 of them clutter every frame."""
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    return opt



def intrinsics(model, cam_name, width, height):
    """MJCF fovy is the vertical field of view, so fy follows from the height."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fovy = np.radians(model.cam_fovy[cid])
    fy = (height / 2) / np.tan(fovy / 2)
    return np.array([[fy, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])


def point_cloud(depth, K, clip):
    """Back-project a depth image into camera-frame points (x right, y down, z fwd)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.copy()
    ok = np.isfinite(z) & (z > 0.05) & (z < clip)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x[ok], y[ok], z[ok]], axis=1)


def colourise(depth, clip):
    d = np.clip(depth, 0, clip) / clip
    d[~np.isfinite(depth)] = 1.0
    t = 1.0 - d                                   # near = bright
    rgb = np.stack([t * 0.25 + 0.05, t * 0.85, t * 0.95], axis=-1)
    return (rgb * 255).astype(np.uint8)


def run(kind):
    cfg = RIGS[kind]
    model = mujoco.MjModel.from_xml_path(str(MODELS / cfg["scene"]))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for name, val in cfg["ctrl"].items():
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = val
    for _ in range(int(1.2 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    with mujoco.Renderer(model, H, W) as r:
        r.update_scene(data, cfg["color"], scene_opt())
        rgb = r.render()
        r.enable_depth_rendering()
        r.update_scene(data, cfg["depth"], scene_opt())
        depth = r.render().astype(np.float64)
        r.disable_depth_rendering()

    K = intrinsics(model, cfg["depth"], W, H)
    pts = point_cloud(depth, K, cfg["clip"])
    near = depth[np.isfinite(depth) & (depth > 0.05)]

    print(f"{kind}: colour {rgb.shape[1]}x{rgb.shape[0]} from '{cfg['color']}', "
          f"depth from '{cfg['depth']}'")
    print(f"  intrinsics fx=fy={K[0,0]:.1f} px, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    print(f"  depth range {near.min():.3f} - {near.max():.3f} m, "
          f"{len(pts)} points inside {cfg['clip']:.0f} m")

    out = f"rgbd_{kind}.png"
    Image.fromarray(np.hstack([rgb, colourise(depth, cfg["clip"])])).save(out)
    np.save(f"cloud_{kind}.npy", pts.astype(np.float32))
    print(f"  wrote {out} and cloud_{kind}.npy")


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    for k in (sys.argv[1:] or ["manipulator", "car"]):
        run(k)
