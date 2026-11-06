#!/usr/bin/env python3
"""
Generates urdf/auriga_car.urdf for the AURIGA self-driving car.

Ackermann front axle (two steer joints + four wheel joints), a pan/tilt camera
mast carrying a D435-style RGB-D bar, three forward ToF sensors, and a
3D-printed vintage-pickup body: ribbed side panels, arched fenders with LED
strips over them, a rear cargo box, and the AURIGA nameplate on both fenders.

Body internals are intentionally left out - a flat electronics deck stands in
for the wiring so the model stays cheap to render and collide.
"""

import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "urdf", "auriga_car.urdf")
MESH_PREFIX = "package://auriga_car_description/meshes"

# ---------------------------------------------------------------- dimensions
WB = 0.260           # wheelbase
TRACK = 0.196        # track width (wheel centre to wheel centre)
WH_R, WH_W = 0.048, 0.034
RIM_R, LIP_R, HUB_R = 0.031, 0.039, 0.011
N_SPOKES = 10

FRAME_L, FRAME_W, FRAME_H = 0.400, 0.150, 0.026
NOSE_L, NOSE_W, NOSE_H = 0.082, 0.186, 0.086      # grille block
CARGO_L, CARGO_W, CARGO_H = 0.150, 0.178, 0.118   # rear box
SIDE_L, SIDE_H = 0.168, 0.056                     # ribbed side panel

FENDER_R = 0.068     # arch radius over each wheel
FENDER_T = 0.013     # arch thickness
FENDER_W = 0.028
LED_R = 0.076        # LED strip rides just outside the arch
N_ARC = 11

MAST_H = 0.110
CAM_L, CAM_W, CAM_H = 0.024, 0.090, 0.024

BASE_Z = WH_R        # base_link sits at axle height
FX, RX = WB / 2, -WB / 2
WY = TRACK / 2

LABEL_L, LABEL_H = 0.070, 0.022   # matches auriga_nameplate.dae


# ---------------------------------------------------------------- helpers
def box_inertia(m, x, y, z):
    return (m * (y * y + z * z) / 12, m * (x * x + z * z) / 12,
            m * (x * x + y * y) / 12)


def cyl_inertia(m, r, h, axis="z"):
    ir, ia = m * (3 * r * r + h * h) / 12, m * r * r / 2
    return {"x": (ia, ir, ir), "y": (ir, ia, ir), "z": (ir, ir, ia)}[axis]


def inertial(m, i, xyz="0 0 0"):
    return (f'    <inertial>\n'
            f'      <origin xyz="{xyz}" rpy="0 0 0"/>\n'
            f'      <mass value="{m:.4f}"/>\n'
            f'      <inertia ixx="{i[0]:.7f}" ixy="0" ixz="0" '
            f'iyy="{i[1]:.7f}" iyz="0" izz="{i[2]:.7f}"/>\n'
            f'    </inertial>')


def vis(geom, material, xyz="0 0 0", rpy="0 0 0"):
    return (f'    <visual>\n      <origin xyz="{xyz}" rpy="{rpy}"/>\n'
            f'      <geometry>{geom}</geometry>\n'
            f'      <material name="{material}"/>\n    </visual>')


def col(geom, xyz="0 0 0", rpy="0 0 0"):
    return (f'    <collision>\n      <origin xyz="{xyz}" rpy="{rpy}"/>\n'
            f'      <geometry>{geom}</geometry>\n    </collision>')


def gbox(x, y, z):
    return f'<box size="{x:.4f} {y:.4f} {z:.4f}"/>'


def gcyl(r, l):
    return f'<cylinder radius="{r:.4f}" length="{l:.4f}"/>'


def joint(name, jtype, parent, child, xyz, rpy="0 0 0", axis=None, limit=None,
          damping=0.02, friction=0.01):
    s = [f'  <joint name="{name}" type="{jtype}">',
         f'    <parent link="{parent}"/>',
         f'    <child link="{child}"/>',
         f'    <origin xyz="{xyz}" rpy="{rpy}"/>']
    if axis:
        s.append(f'    <axis xyz="{axis}"/>')
    if limit:
        lo, hi, eff, vel = limit
        s.append(f'    <limit effort="{eff}" velocity="{vel}"/>' if lo is None
                 else f'    <limit lower="{lo:.4f}" upper="{hi:.4f}" '
                      f'effort="{eff}" velocity="{vel}"/>')
    if jtype != "fixed":
        s.append(f'    <dynamics damping="{damping}" friction="{friction}"/>')
    s.append("  </joint>")
    return "\n".join(s)


def arc(cx, cz, y, radius, seg_l, seg_w, seg_t, material,
        a0=18.0, a1=162.0, n=N_ARC):
    """Ring of tangent-aligned slabs -> a wheel arch or an LED strip.

    A box rotated about +Y by -(pi/2 + theta) has its long axis on the tangent
    at angle theta, so the segments meet cleanly around the arch.
    """
    out = []
    for i in range(n):
        th = math.radians(a0 + (a1 - a0) * i / (n - 1))
        px, pz = cx + radius * math.cos(th), cz + radius * math.sin(th)
        out.append(vis(gbox(seg_l, seg_w, seg_t), material,
                       xyz=f"{px:.4f} {y:.4f} {pz:.4f}",
                       rpy=f"0 {-(math.pi / 2 + th):.4f} 0"))
    return out


L, G = [], []


def add_link(name, body):
    L.append(f'  <link name="{name}">\n' + "\n".join(body) + "\n  </link>")


def gz(link, mat):
    G.append(f'  <gazebo reference="{link}">\n    <material>{mat}</material>\n  </gazebo>')


# ---------------------------------------------------------------- base
L.append('  <link name="base_footprint"/>')
L.append(joint("base_footprint_joint", "fixed", "base_footprint", "base_link",
               f"0 0 {BASE_Z:.4f}"))

body = [inertial(2.6, box_inertia(2.6, FRAME_L, FRAME_W, 0.12), xyz="-0.02 0 0.03")]

# ladder frame + floor pan
body += [vis(gbox(FRAME_L, FRAME_W, FRAME_H), "auriga_navy", xyz="-0.012 0 -0.006"),
         col(gbox(FRAME_L, FRAME_W, 0.070), xyz="-0.012 0 0.012")]

# nose: grille block with horizontal slats
body += [vis(gbox(NOSE_L, NOSE_W, NOSE_H), "auriga_navy",
             xyz=f"{FX + 0.052:.4f} 0 0.030"),
         col(gbox(NOSE_L, NOSE_W, NOSE_H), xyz=f"{FX + 0.052:.4f} 0 0.030")]
for k in range(5):
    body.append(vis(gbox(0.006, 0.120, 0.006), "auriga_navy_dark",
                    xyz=f"{FX + 0.094:.4f} 0 {0.006 + 0.014 * k:.4f}"))
# bumper
body.append(vis(gbox(0.016, NOSE_W + 0.006, 0.020), "auriga_navy",
                xyz=f"{FX + 0.098:.4f} 0 -0.012"))
# headlight bezels
for s in (1, -1):
    body.append(vis(gcyl(0.012, 0.012), "auriga_navy_dark",
                    xyz=f"{FX + 0.092:.4f} {s * 0.076:.4f} 0.052", rpy="0 1.5708 0"))

# ribbed side panels between the fenders
for s in (1, -1):
    y = s * (FRAME_W / 2 + 0.006)
    body.append(vis(gbox(SIDE_L, 0.012, SIDE_H), "auriga_navy",
                    xyz=f"0.021 {y:.4f} 0.020"))
    for k in range(5):                                   # vertical ribs, rear half
        body.append(vis(gbox(0.006, 0.005, SIDE_H - 0.014), "auriga_navy_dark",
                        xyz=f"{-0.052 + 0.018 * k:.4f} {y + s * 0.007:.4f} 0.020"))

# rear cargo box + roof cap
body += [vis(gbox(CARGO_L, CARGO_W, CARGO_H), "auriga_navy",
             xyz=f"{RX - 0.036:.4f} 0 {CARGO_H / 2 + 0.006:.4f}"),
         col(gbox(CARGO_L, CARGO_W, CARGO_H), xyz=f"{RX - 0.036:.4f} 0 {CARGO_H / 2 + 0.006:.4f}"),
         vis(gbox(CARGO_L - 0.030, CARGO_W - 0.014, 0.016), "auriga_navy",
             xyz=f"{RX - 0.044:.4f} 0 {CARGO_H + 0.012:.4f}"),
         # tail lights
         vis(gbox(0.006, 0.016, 0.026), "auriga_red",
             xyz=f"{RX - 0.112:.4f} 0.072 0.052"),
         vis(gbox(0.006, 0.016, 0.026), "auriga_red",
             xyz=f"{RX - 0.112:.4f} -0.072 0.052")]

# fenders + LED strips over all four wheels
for x, tag in ((FX, "f"), (RX, "r")):
    for s in (1, -1):
        y = s * (WY - 0.010)
        body += arc(x, 0.0, y, FENDER_R, 0.026, FENDER_W, FENDER_T, "auriga_navy")
        if tag == "f":   # LED strips only on the front arches, as on the robot
            body += arc(x, 0.0, y + s * 0.012, LED_R, 0.020, 0.008, 0.006,
                        "auriga_led", a0=26, a1=154, n=9)

# AURIGA nameplate on both front fender panels
plate = f'<mesh filename="{MESH_PREFIX}/auriga_nameplate.dae"/>'
for s, rpy_ in ((1, "-1.5708 0 0"), (-1, "1.5708 3.14159 0")):
    body.append(vis(plate, "auriga_navy",
                    xyz="0.068 " + f"{s * (FRAME_W / 2 + 0.014):.4f} 0.030",
                    rpy=rpy_))

# electronics deck (stand-in for the real wiring loom)
body += [vis(gbox(0.170, 0.140, 0.010), "auriga_carbon", xyz="0.010 0 0.052"),
         vis(gbox(0.070, 0.048, 0.014), "auriga_pcb", xyz="0.048 0.030 0.066"),
         vis(gbox(0.052, 0.040, 0.012), "auriga_carbon", xyz="-0.030 -0.028 0.065"),
         vis(gbox(0.030, 0.062, 0.022), "auriga_carbon", xyz="-0.052 0.020 0.070")]

# three forward ToF boards on the grille shoulder
for y in (0.062, 0.0, -0.062):
    body += [vis(gbox(0.008, 0.014, 0.020), "auriga_pcb_red",
                 xyz=f"{FX + 0.048:.4f} {y:.4f} 0.082"),
             vis(gcyl(0.004, 0.004), "auriga_lens",
                 xyz=f"{FX + 0.053:.4f} {y:.4f} 0.084", rpy="0 1.5708 0")]

add_link("base_link", body)
gz("base_link", "Gazebo/Blue")

# ---------------------------------------------------------------- wheels
def wheel_body(name):
    b = [inertial(0.09, cyl_inertia(0.09, WH_R, WH_W, axis="y")),
         vis(gcyl(WH_R, WH_W), "auriga_tire", rpy="1.5708 0 0"),
         vis(gcyl(LIP_R, WH_W + 0.003), "auriga_whitewall", rpy="1.5708 0 0"),
         vis(gcyl(RIM_R, WH_W + 0.005), "auriga_rim", rpy="1.5708 0 0"),
         vis(gcyl(HUB_R, WH_W + 0.007), "auriga_chrome", rpy="1.5708 0 0")]
    for k in range(N_SPOKES):                    # BBS-style split spokes
        a = 2 * math.pi * k / N_SPOKES
        r = RIM_R * 0.6
        b.append(vis(gbox(RIM_R * 0.95, WH_W + 0.006, 0.004), "auriga_rim_dark",
                     xyz=f"{r * math.cos(a):.4f} 0 {r * math.sin(a):.4f}",
                     rpy=f"0 {-a:.4f} 0"))
    b.append(col(gcyl(WH_R, WH_W), rpy="1.5708 0 0"))
    return b


STEER_LIM = 0.55   # ~31 deg
for s, side in ((1, "left"), (-1, "right")):
    # steering knuckle
    add_link(f"front_{side}_steer_link", [
        inertial(0.03, box_inertia(0.03, 0.02, 0.02, 0.04)),
        vis(gbox(0.020, 0.016, 0.040), "auriga_carbon"),
    ])
    L.append(joint(f"front_{side}_steer_joint", "revolute", "base_link",
                   f"front_{side}_steer_link",
                   f"{FX:.4f} {s * WY:.4f} 0", axis="0 0 1",
                   limit=(-STEER_LIM, STEER_LIM, 8.0, 4.0),
                   damping=0.08, friction=0.05))
    add_link(f"front_{side}_wheel_link", wheel_body(f"front_{side}"))
    L.append(joint(f"front_{side}_wheel_joint", "continuous",
                   f"front_{side}_steer_link", f"front_{side}_wheel_link",
                   "0 0 0", axis="0 1 0", limit=(None, None, 6.0, 60.0)))

    add_link(f"rear_{side}_wheel_link", wheel_body(f"rear_{side}"))
    L.append(joint(f"rear_{side}_wheel_joint", "continuous", "base_link",
                   f"rear_{side}_wheel_link", f"{RX:.4f} {s * WY:.4f} 0",
                   axis="0 1 0", limit=(None, None, 6.0, 60.0)))

for n in ("front_left", "front_right", "rear_left", "rear_right"):
    gz(f"{n}_wheel_link", "Gazebo/DarkGrey")
    G.append(f"""  <gazebo reference="{n}_wheel_link">
    <mu1>1.3</mu1>
    <mu2>1.1</mu2>
    <kp>2e6</kp>
    <kd>15.0</kd>
    <minDepth>0.0008</minDepth>
  </gazebo>""")

# ---------------------------------------------------------------- camera mast
add_link("mast_link", [
    inertial(0.06, box_inertia(0.06, 0.026, 0.026, MAST_H), xyz=f"0 0 {MAST_H / 2:.4f}"),
    vis(gbox(0.034, 0.040, 0.020), "auriga_navy", xyz="0 0 0.008"),
    vis(gbox(0.020, 0.024, MAST_H), "auriga_navy", xyz=f"0 0 {MAST_H / 2 + 0.014:.4f}"),
    col(gbox(0.024, 0.028, MAST_H), xyz=f"0 0 {MAST_H / 2:.4f}"),
])
L.append(joint("mast_joint", "fixed", "base_link", "mast_link",
               f"{RX + 0.062:.4f} 0 {CARGO_H * 0.30:.4f}"))
gz("mast_link", "Gazebo/Blue")

add_link("camera_pan_link", [
    inertial(0.02, cyl_inertia(0.02, 0.012, 0.020)),
    vis(gcyl(0.012, 0.020), "auriga_chrome"),
])
L.append(joint("camera_pan_joint", "revolute", "mast_link", "camera_pan_link",
               f"0 0 {MAST_H + 0.030:.4f}", axis="0 0 1",
               limit=(-1.5708, 1.5708, 3.0, 2.0), damping=0.05))

add_link("camera_tilt_link", [
    inertial(0.03, box_inertia(0.03, 0.02, 0.05, 0.03)),
    vis(gbox(0.018, 0.046, 0.024), "auriga_navy", xyz="0 0 0.014"),
    vis(gcyl(0.010, 0.030), "auriga_chrome", xyz="0 0 0.004", rpy="1.5708 0 0"),
])
L.append(joint("camera_tilt_joint", "revolute", "camera_pan_link",
               "camera_tilt_link", "0 0 0.014", axis="0 1 0",
               limit=(-0.7854, 0.5236, 3.0, 2.0), damping=0.05))

# D435-style RGB-D bar
add_link("camera_link", [
    inertial(0.05, box_inertia(0.05, CAM_L, CAM_W, CAM_H)),
    vis(gbox(CAM_L, CAM_W, CAM_H), "auriga_white"),
    col(gbox(CAM_L, CAM_W, CAM_H)),
    vis(gbox(CAM_L - 0.004, CAM_W - 0.014, 0.014), "auriga_lens", xyz="0.003 0 0.002"),
    vis(gcyl(0.007, 0.006), "auriga_lens", xyz="0.012 0.031 0.002", rpy="0 1.5708 0"),
    vis(gcyl(0.006, 0.006), "auriga_lens", xyz="0.012 0.010 0.002", rpy="0 1.5708 0"),
    vis(gcyl(0.005, 0.006), "auriga_ir", xyz="0.012 -0.008 0.002", rpy="0 1.5708 0"),
    vis(gcyl(0.007, 0.006), "auriga_lens", xyz="0.012 -0.031 0.002", rpy="0 1.5708 0"),
])
L.append(joint("camera_mount_joint", "fixed", "camera_tilt_link", "camera_link",
               "0.004 0 0.030"))
gz("camera_link", "Gazebo/White")

L.append('  <link name="camera_optical_frame"/>')
L.append(joint("camera_optical_joint", "fixed", "camera_link",
               "camera_optical_frame", "0 0 0", rpy="-1.5708 0 -1.5708"))

# ---------------------------------------------------------------- gazebo
G.append("""  <!-- Ackermann drive: rear wheels propel, front knuckles steer.
       The plugin solves the steering trapezoid itself, so publish one
       Twist and it splits the inner/outer angles for you. -->
  <gazebo>
    <plugin name="ackermann_drive" filename="libgazebo_ros_ackermann_drive.so">
      <ros>
        <namespace>/auriga_car</namespace>
        <remapping>cmd_vel:=cmd_vel</remapping>
        <remapping>odom:=odom</remapping>
        <remapping>distance:=distance</remapping>
      </ros>
      <update_rate>100.0</update_rate>

      <front_left_joint>front_left_wheel_joint</front_left_joint>
      <front_right_joint>front_right_wheel_joint</front_right_joint>
      <rear_left_joint>rear_left_wheel_joint</rear_left_joint>
      <rear_right_joint>rear_right_wheel_joint</rear_right_joint>
      <left_steering_joint>front_left_steer_joint</left_steering_joint>
      <right_steering_joint>front_right_steer_joint</right_steering_joint>

      <max_steer>0.55</max_steer>
      <max_steering_angle>0.55</max_steering_angle>
      <max_speed>2.0</max_speed>

      <left_steering_pid_gain>6.0 0.0 0.4</left_steering_pid_gain>
      <left_steering_i_range>0 0</left_steering_i_range>
      <right_steering_pid_gain>6.0 0.0 0.4</right_steering_pid_gain>
      <right_steering_i_range>0 0</right_steering_i_range>
      <linear_velocity_pid_gain>25.0 0.0 1.0</linear_velocity_pid_gain>
      <linear_velocity_i_range>0 0</linear_velocity_i_range>

      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>false</publish_wheel_tf>
      <publish_distance>true</publish_distance>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_footprint</robot_base_frame>
    </plugin>
  </gazebo>

  <gazebo>
    <plugin name="joint_state_pub" filename="libgazebo_ros_joint_state_publisher.so">
      <ros><namespace>/auriga_car</namespace></ros>
      <update_rate>50</update_rate>
      <joint_name>front_left_steer_joint</joint_name>
      <joint_name>front_right_steer_joint</joint_name>
      <joint_name>front_left_wheel_joint</joint_name>
      <joint_name>front_right_wheel_joint</joint_name>
      <joint_name>rear_left_wheel_joint</joint_name>
      <joint_name>rear_right_wheel_joint</joint_name>
      <joint_name>camera_pan_joint</joint_name>
      <joint_name>camera_tilt_joint</joint_name>
    </plugin>
  </gazebo>

  <!-- Mast RGB-D -->
  <gazebo reference="camera_link">
    <sensor name="mast_d435" type="depth">
      <update_rate>30</update_rate>
      <camera>
        <horizontal_fov>1.5184</horizontal_fov>
        <image><width>848</width><height>480</height><format>R8G8B8</format></image>
        <clip><near>0.08</near><far>10.0</far></clip>
      </camera>
      <plugin name="mast_camera" filename="libgazebo_ros_camera.so">
        <ros><namespace>/auriga_car</namespace></ros>
        <camera_name>mast_camera</camera_name>
        <frame_name>camera_optical_frame</frame_name>
        <min_depth>0.08</min_depth>
        <max_depth>10.0</max_depth>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Three forward ToF rangefinders on the grille -->
  <gazebo reference="base_link">
    <sensor name="tof_array" type="ray">
      <pose>0.183 0 0.082 0 0 0</pose>
      <update_rate>20</update_rate>
      <ray>
        <scan><horizontal><samples>3</samples><resolution>1</resolution>
          <min_angle>-0.35</min_angle><max_angle>0.35</max_angle></horizontal></scan>
        <range><min>0.03</min><max>1.2</max><resolution>0.002</resolution></range>
      </ray>
      <plugin name="tof" filename="libgazebo_ros_ray_sensor.so">
        <ros><namespace>/auriga_car</namespace><remapping>~/out:=tof</remapping></ros>
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>base_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>""")

MATERIALS = """  <material name="auriga_navy"><color rgba="0.118 0.204 0.404 1.0"/></material>
  <material name="auriga_navy_dark"><color rgba="0.071 0.129 0.271 1.0"/></material>
  <material name="auriga_carbon"><color rgba="0.086 0.090 0.098 1.0"/></material>
  <material name="auriga_tire"><color rgba="0.086 0.086 0.094 1.0"/></material>
  <material name="auriga_whitewall"><color rgba="0.906 0.910 0.914 1.0"/></material>
  <material name="auriga_rim"><color rgba="0.541 0.086 0.114 1.0"/></material>
  <material name="auriga_rim_dark"><color rgba="0.400 0.055 0.078 1.0"/></material>
  <material name="auriga_chrome"><color rgba="0.749 0.769 0.800 1.0"/></material>
  <material name="auriga_led"><color rgba="0.949 0.961 0.984 1.0"/></material>
  <material name="auriga_white"><color rgba="0.929 0.933 0.941 1.0"/></material>
  <material name="auriga_lens"><color rgba="0.055 0.075 0.110 1.0"/></material>
  <material name="auriga_ir"><color rgba="0.180 0.086 0.220 1.0"/></material>
  <material name="auriga_red"><color rgba="0.702 0.106 0.106 1.0"/></material>
  <material name="auriga_pcb"><color rgba="0.055 0.278 0.180 1.0"/></material>
  <material name="auriga_pcb_red"><color rgba="0.612 0.098 0.114 1.0"/></material>"""

doc = f"""<?xml version="1.0"?>
<!-- AURIGA self-driving car: Ackermann chassis + pan/tilt RGB-D mast + ToF array.
     GENERATED by generate_urdf.py - edit that file, not this one. -->
<robot name="auriga_car">

{MATERIALS}

{chr(10).join(L)}

{chr(10).join(G)}

</robot>
"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(doc)
print(f"wrote {OUT} ({len(doc)} bytes)")
