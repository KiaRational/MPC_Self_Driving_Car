# auriga_mujoco

MJCF models of both AURIGA robots, generated from the same dimensions as the two
URDF packages.

```
auriga_mujoco/
├── gen_mjcf.py                # source of truth - regenerates both XMLs + the plate meshes
├── auriga_manipulator.xml     # mecanum base + 4-DOF arm + parallel gripper
├── auriga_car.xml             # Ackermann car + pan/tilt RGB-D mast + ToF array
├── scene_manipulator.xml      # floor, sky, lights, two graspable blocks
├── scene_car.xml              # floor, sky, lights, cone lane, wall
├── demo_manipulator.py        # drives a square, cycles arm poses
├── demo_car.py                # runs the cone lane, stops on the ToF return
├── realsense.py               # RGB + depth + point cloud from the D435 cameras
├── lidar.py                   # 360 / multi-beam scans via mj_multiRay
├── render.py                  # headless preview stills (OSMesa/EGL)
└── assets/                    # nameplate textures + UV-mapped plate meshes
```

Load a scene, not a robot: `scene_*.xml` includes the robot and adds the world.

```bash
pip install mujoco
python -m mujoco.viewer --mjcf=scene_manipulator.xml
python demo_car.py                 # or --headless
MUJOCO_GL=osmesa python render.py  # stills without a display
```

## Manipulator

| | |
|---|---|
| DOF | 53 qpos / 52 qvel (free base, 4 wheels, 36 rollers, 6 arm) |
| Actuators | 4 wheel velocity, 4 arm position, 1 gripper position |
| Sensors | joint positions, grasp-site pose, IMU, velocimeter, fingertip touch |
| Keyframe | `photo_pose` |

The mecanum wheels are **real**: nine free-spinning capsule rollers per wheel at
±45°, and the base slides sideways because of the contact geometry rather than a
plugin. Measured against commanded body twists over 2 s:

| command | measured |
|---|---|
| vx = 0.40 m/s | 0.40 m/s forward, no yaw drift |
| vy = 0.40 m/s | 0.39 m/s sideways, 0.01 rad drift |
| ωz = 1.50 rad/s | 1.49 rad/s |

The mixer that produces those wheel rates:

```
ω_fl = (vx − vy − (lx+ly)·ωz) / r
ω_fr = (vx + vy + (lx+ly)·ωz) / r
ω_rl = (vx + vy − (lx+ly)·ωz) / r
ω_rr = (vx − vy + (lx+ly)·ωz) / r      lx+ly = 0.378 m,  r = 0.065 m
```

Roller chirality was fixed by testing against this mixer, and the correction was
pushed back into the URDF package — the original URDF had it mirrored, which
would have strafed the wrong way.

Two things to know if you edit the wheels:

- Rollers carry `contype="0" conaffinity="1"`. They hit the floor and never each
  other. Without that, neighbouring capsules jam and the wheel behaves like a
  solid tyre — the base then refuses to strafe and drifts in yaw instead.
- Raising `N_ROLLERS` smooths the ride over the contact gaps. Past about nine the
  capsules start to overlap visually, which is harmless since they cannot collide.

Both jaws are driven by one actuator (`gripper`) with an equality constraint
coupling the second finger. The jaws open to 66 mm, so the 35 mm blocks in the
scene are a comfortable grasp.

## Car

| | |
|---|---|
| DOF | 15 qpos / 14 qvel (free base, 2 steer, 4 wheels, pan, tilt) |
| Actuators | 2 steering position, 2 rear-wheel velocity, pan, tilt |
| Sensors | steer angles, wheel rates, IMU, velocimeter, 3 rangefinders |
| Keyframe | `home` |

Nothing solves Ackermann for you here — each knuckle takes its own target, which
is the point of modelling the axle rather than using a drive plugin. Measured
turn radius against the geometric command:

| δ (virtual centre) | R commanded | R measured |
|---|---|---|
| 0.25 rad | 1.02 m | 1.11 m |
| 0.55 rad (full lock) | 0.42 m | 0.51 m |

The gap is tyre slip, not a modelling error; it grows with speed and steering
angle the way understeer should.

The three ToF rangefinders carry `cutoff="1.2"` to match a real module's range.
A rangefinder returns **−1 when the ray hits nothing**, so treat only a positive
short return as an obstacle — `demo_car.py` shows the check, and getting it wrong
parks the car permanently.

## RealSense

MuJoCo has no depth-sensor element. A `<camera>` is only a viewpoint; **depth
comes from the renderer**, so an RGB-D stream is two calls:

```python
renderer.update_scene(data, "wrist_color"); rgb = renderer.render()
renderer.enable_depth_rendering()
renderer.update_scene(data, "wrist_depth"); depth = renderer.render()   # metres
```

Each robot therefore carries two cameras matching the D435's two real streams,
offset by the module's 17.5 mm baseline:

| camera | FOV (MJCF `fovy` is vertical) | resolution |
|---|---|---|
| `wrist_color` / `mast_color` | 42° V (69° H) | 1280x720 |
| `wrist_depth` / `mast_depth` | 58° V (87° H) | 848x480 |

`realsense.py` renders both, derives the pinhole intrinsics from `fovy`
(fx = fy = 433 px at 848x480), back-projects a point cloud and saves it. Note
the offscreen framebuffer defaults to 640x480 — both models set
`<global offwidth="1280" offheight="720"/>` or the renderer refuses these sizes.

What you do **not** get for free: stereo IR speckle, depth noise, holes on shiny
or dark surfaces, or the rolling-shutter behaviour. MuJoCo depth is a clean
z-buffer readout. If you are training something that must survive real D435
data, add noise and dropout yourself.

## Lidar

Both robots now carry a spinning 2D scanner: the manipulator's sits over the
rear guard (12 m range), the car's on the hood (8 m). Two ways to read it:

- **24 `rangefinder` sensors** baked into each model, in `data.sensordata` every
  step. Cheap and always available, but fixed resolution.
- **`lidar.py`**, which casts arbitrarily many rays with `mj_multiRay` on demand —
  `scan2d()` for a 360x1 sweep, `scan3d()` for a multi-beam puck. A 360-ray
  sweep costs one call. The two agree to 0.0000 m, which is the check the script
  prints.

Both robots see their own bodywork, since a real scanner does too — the arm
shadows the manipulator's sweep when raised, and the cargo box blocks the car's
rear sector. `scan2d(..., self_hits=False)` masks geom groups 2 and 3 to get a
world-only sweep.

A rangefinder returns **−1 for a miss**, and `lidar.py` converts that to `inf`.
The same −1 trips up the ToF check in `demo_car.py` if you forget it.

## Quirks worth knowing

- **Wheels are grandchildren of `base_link`**, and MuJoCo only auto-excludes
  direct parent/child contact pairs. Both models carry explicit
  `<contact><exclude>` entries. Without them the car's front tyres sit inside the
  nose collision box, which locks them and caps the car at 0.11 m/s.
- **A free joint's qpos is the body pose in world coordinates**, not an offset
  from the body's `pos`. Keyframes must carry the real ride height or the robot
  starts buried in the floor and launches itself on the first step.
- **Nameplates need ~4 mm of standoff.** At 1 mm the arm box wins the depth test
  and the lettering disappears even though the plate is geometrically outside it.
- The plates are UV-mapped OBJ slabs rather than textured boxes: MuJoCo
  cube-maps a 2D texture onto a box, which crops the artwork. They also carry a
  little `emission`, since both scenes light from above and a nameplate on a
  vertical face would otherwise read as flat black.

## Fidelity

Inertias come from primitive shapes at plausible masses, not CAD. Collision
geometry is coarse — one box or cylinder per link, wheels as plain cylinders
(rollers on the mecanum base). Good for locomotion, planning and RL; replace the
inertias and the finger/pad geometry before trusting it for contact-rich
manipulation.
