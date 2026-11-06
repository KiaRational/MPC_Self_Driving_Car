# auriga_car_description

URDF for the **AURIGA self-driving car** — a 3D-printed vintage-pickup body on an
Ackermann chassis, with a pan/tilt mast carrying a D435-style RGB-D bar and three
forward ToF rangefinders. Reconstructed from a photo, so treat the dimensions as
a starting point: measure the real car and edit `generate_urdf.py`.

```
auriga_car_description/
├── urdf/auriga_car.urdf          # plain URDF (generated) - no xacro needed
├── generate_urdf.py              # source of truth; regenerates the URDF
├── meshes/auriga_nameplate.dae   # AURIGA lettering on both body panels
├── meshes/auriga_nameplate.png
├── launch/gazebo.launch.py
├── launch/display.launch.py      # RViz + joint sliders, no simulator
├── rviz/auriga_car.rviz
└── auriga_car_viewer.html        # standalone 3D preview (open in a browser)
```

## Kinematic tree

```
base_footprint
└── base_link
    ├── front_{left,right}_steer_link   *_steer_joint  (revolute, ±0.55 rad ≈ ±31°)
    │   └── front_{left,right}_wheel_link  *_wheel_joint (continuous, axis Y)
    ├── rear_{left,right}_wheel_link       *_wheel_joint (continuous, axis Y)
    └── mast_link                       (fixed)
        └── camera_pan_link             camera_pan_joint  (revolute, ±90°)
            └── camera_tilt_link        camera_tilt_joint (revolute, -45°..+30°)
                └── camera_link → camera_optical_frame (fixed, REP-103)
```

Wheelbase 260 mm, track 196 mm, wheel radius 48 mm, overall length ~440 mm,
camera lens ~270 mm above the ground.

## Run it

```bash
colcon build --packages-select auriga_car_description
source install/setup.bash

ros2 launch auriga_car_description display.launch.py     # RViz + sliders
ros2 launch auriga_car_description gazebo.launch.py      # Gazebo Classic

# drive: linear.x is speed, angular.z is the steering command
ros2 topic pub /auriga_car/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.6}, angular: {z: 0.4}}"

ros2 topic echo /auriga_car/odom
ros2 topic echo /auriga_car/tof
ros2 run rqt_image_view rqt_image_view /auriga_car/mast_camera/image_raw
```

## Steering notes

The URDF is wired to `gazebo_ros_ackermann_drive`, which drives both rear wheels
and PID-tracks the two steering joints. It solves the steering trapezoid
internally, so you publish a single command and it splits the angles: at full
lock the inner wheel runs ~7° sharper than the outer one. The formula, if you
want to close your own loop instead:

```
R      = L / tan(δ)                 L = 0.260 m,  T = 0.196 m
δ_in   = atan( L / (R − T/2) )
δ_out  = atan( L / (R + T/2) )
```

Minimum turn radius at full lock is about 0.44 m to the centreline.

If you'd rather run `ros2_control`, delete the `ackermann_drive` plugin block and
add `position_controllers` on the two steer joints plus `velocity_controllers` on
the rear wheels — the joint names above are already limit-tagged for it. The
`ackermann_steering_controller` in recent `ros2_controllers` releases also maps
onto this joint layout directly.

## Notes

- Body internals are deliberately omitted. A flat carbon-coloured deck with two
  small board shapes stands in for the wiring loom, so the mesh count stays low
  and collision checking stays cheap.
- Collision geometry is one coarse box for the chassis, one for the cargo box,
  one for the mast, and a cylinder per wheel. The fender arches and LED strips
  are visual-only (they're built from ~90 small slabs around each arch).
- The LED strips use an emissive material — they'll glow in Gazebo's PBR
  rendering but stay flat white in RViz, which ignores emission.
- Tyre friction is set to `mu1 1.3 / mu2 1.1`. Lower `mu2` if you want the rear
  to break away for drift experiments.
- `camera_optical_frame` follows REP-103, so depth clouds land in the right
  orientation without a static transform hack.
