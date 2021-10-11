#!/usr/bin/env python3
#-*-coding: utf8-*-
import rospy
from cv_bridge import CvBridge
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from geometry_msgs.msg import Twist, TransformStamped

import numpy as np
import quaternion
import habitat_sim




class HabitatSimNode:
    def __init__(self):
        rospy.init_node("habitat_sim")
        # TODO: use simtime (/clock topic?)
        self._rate = rospy.Rate(rospy.get_param("~sim/rate"))
        self._cmd_vel_sub = rospy.Subscriber("~cmd_vel", Twist, self._on_cmd_vel)
        self._last_cmd_vel = rospy.Time.now()
        self._cmd_timeout = rospy.Duration(rospy.get_param("~agent/cmd_timeout"))
        self._has_cmd = False
        self._tf_brdcast = TransformBroadcaster()
        self._scan_pub = rospy.Publisher("~scan", LaserScan, queue_size=1)
        self._rgb_pub = rospy.Publisher("~camera/rgb/image_raw", Image, queue_size=1)
        self._depth_pub = rospy.Publisher("~camera/depth/image_raw", Image, queue_size=1)
        self._cv_bridge = CvBridge()

        self._sensor_specs = []
        self._static_tfs = []
        rospy.loginfo("Setting up scan sensor")
        self._init_scan()
        rospy.loginfo("Setting up rgbd sensor")
        self._init_rgbd()
        rospy.loginfo("Starting simulator")
        self._init_sim()

    def loop(self):
        self._broadcast_static_tf()
        while not rospy.is_shutdown():
            self._broadcast_tf()
            self._publish_scan()
            self._publish_rgbd()
            self._step_sim()
            self._rate.sleep()

    def _init_scan(self):
        x = rospy.get_param("~scan/position/x")
        y = rospy.get_param("~scan/position/y")
        z = rospy.get_param("~scan/position/z")
        r_min = rospy.get_param("~scan/range/min")
        r_max = rospy.get_param("~scan/range/max")
        n_rays = rospy.get_param("~scan/num_rays")

        self._scan_msg = LaserScan()
        self._scan_msg.header.frame_id = "scan"
        self._scan_msg.angle_min = -np.pi
        self._scan_msg.angle_max = np.pi
        self._scan_msg.angle_increment = 2 * np.pi / n_rays
        self._scan_msg.time_increment = 0.0
        self._scan_msg.range_min = r_min
        self._scan_msg.range_max = r_max

        tf = TransformStamped()
        tf.header.stamp = rospy.Time.now()
        tf.header.frame_id = "base_footprint"
        tf.child_frame_id = "scan"
        tf.transform.translation.x = x
        tf.transform.translation.x = y
        tf.transform.translation.x = z
        tf.transform.rotation.w = 1
        self._static_tfs.append(tf)

        n_rays_per_scan = n_rays // 4
        for prefix, pan in zip(("br", "fr", "fl", "bl"),
                               (-0.75 * np.pi, -0.25 * np.pi, 0.25 * np.pi, 0.75 * np.pi)):
            spec = habitat_sim.sensor.SensorSpec()
            spec.uuid = f"{prefix}_scan"
            spec.sensor_type = habitat_sim.sensor.SensorType.DEPTH
            spec.position = [-y, z, -x]
            spec.orientation = [0.0, pan, 0.0]
            spec.resolution = [1, n_rays_per_scan]
            spec.parameters["hfov"] = "90"
            self._sensor_specs.append(spec)
        d_to_rho = np.sqrt(np.linspace(-0.5, 0.5, n_rays_per_scan)**2 + 1)
        self._scan_d_to_rho = np.tile(d_to_rho, (4,))

    def _init_rgbd(self):
        for sensor, sensor_type in (("rgb", habitat_sim.sensor.SensorType.COLOR),
                                    ("depth", habitat_sim.sensor.SensorType.DEPTH)):
            x = rospy.get_param(f"~{sensor}/position/x")
            y = rospy.get_param(f"~{sensor}/position/y")
            z = rospy.get_param(f"~{sensor}/position/z")
            tilt = np.radians(rospy.get_param(f"~{sensor}/orientation/tilt"))
            h = rospy.get_param(f"~{sensor}/height")
            w = rospy.get_param(f"~{sensor}/width")
            hfov = rospy.get_param(f"~{sensor}/hfov")
            f = 0.5 * w / np.tan(0.5 * hfov)

            info_pub = rospy.Publisher(f"~camera/{sensor}/camera_info", CameraInfo,
                                       queue_size=1, latch=True)
            cam_info = CameraInfo()
            cam_info.header.stamp = rospy.Time.now()
            cam_info.header.frame_id = f"camera/{sensor}/optical_frame"
            cam_info.height = h
            cam_info.width = w
            cam_info.distortion_model = "plumb_bob"
            cam_info.D = [.0, .0, .0, .0, .0]
            cam_info.K = [f, 0, 0.5 * w,
                          0, f, 0.5 * h,
                          0, 0,       1]
            cam_info.R = [1, 0, 0,
                          0, 1, 0,
                          0, 0, 1]
            cam_info.P = [f, 0, 0.5 * w, 0,
                          0, f, 0.5 * h, 0,
                          0, 0,       1, 0]
            info_pub.publish(cam_info)

            tf = TransformStamped()
            tf.header.stamp = rospy.Time.now()
            tf.header.frame_id = "base_footprint"
            tf.child_frame_id = f"camera/{sensor}/optical_frame"
            tf.transform.translation.x = x
            tf.transform.translation.x = y
            tf.transform.translation.x = z
            tf.transform.rotation.y = np.sin(0.5 * tilt)
            tf.transform.rotation.w = np.cos(0.5 * tilt)
            self._static_tfs.append(tf)

            spec = habitat_sim.sensor.SensorSpec()
            spec.uuid = sensor
            spec.sensor_type = sensor_type
            spec.position = [-y, z, -x]
            spec.orientation = [-tilt, 0, 0]
            spec.resolution = [h, w]
            spec.parameters["hfov"] = str(hfov)
            self._sensor_specs.append(spec)

    def _init_sim(self):
        sim_cfg = habitat_sim.sim.SimulatorConfiguration()
        sim_cfg.allow_sliding = rospy.get_param("~sim/allow_sliding")
        sim_cfg.scene_id = rospy.get_param("~sim/scene_id")

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = rospy.get_param("~agent/height")
        agent_cfg.radius = rospy.get_param("~agent/radius")
        agent_cfg.sensor_specifications = self._sensor_specs

        cfg = habitat_sim.simulator.Configuration(sim_cfg, [agent_cfg])
        self._sim = habitat_sim.simulator.Simulator(cfg)

        state = habitat_sim.agent.AgentState()
        if rospy.get_param("~agent/start_position/random"):
            state.position = self._sim.pathfinder.get_random_navigable_point()
        else:
            state.position[0] = -rospy.get_param("~agent/start_position/y")
            state.position[1] = rospy.get_param("~agent/start_position/z")
            state.position[2] = -rospy.get_param("~agent/start_position/x")
        if rospy.get_param("~agent/start_orientation/random"):
            yaw = 2 * np.pi * np.random.random()
        else:
            yaw = np.radians(rospy.get_param("~agent/start_orientation/yaw"))
        state.rotation = np.quaternion(np.cos(0.5 * yaw), 0, np.sin(0.5 * yaw), 0)
        self._sim.get_agent(0).set_state(state, is_initial=True)
        self._last_obs = self._sim.get_sensor_observations()

        self._vel_ctrl = habitat_sim.physics.VelocityControl()
        self._vel_ctrl.controlling_lin_vel = True
        self._vel_ctrl.lin_vel_is_local = True
        self._vel_ctrl.controlling_ang_vel = True
        self._vel_ctrl.ang_vel_is_local = True

        mngr = self._sim.get_object_template_manager()
        tmpl_id = mngr.get_template_ID_by_handle(*mngr.get_template_handles('cylinderSolid'))
        self._agent_obj_id = self._sim.add_object(tmpl_id, self._sim.get_agent(0).scene_node)

    def _on_cmd_vel(self, msg):
        self._vel_ctrl.linear_velocity.z = -msg.linear.x
        self._vel_ctrl.angular_velocity.y = msg.angular.z
        self._last_cmd_vel = rospy.Time.now()
        self._has_cmd = True

    def _broadcast_static_tf(self):
        static_tf_brdcast = StaticTransformBroadcaster()
        static_tf_brdcast.sendTransform(self._static_tfs)

    def _broadcast_tf(self):
        state = self._sim.get_agent(0).get_state()
        init_state = self._sim.get_agent(0).initial_state

        rel_pos = (init_state.rotation.conj()
                   * np.quaternion(0, *(state.position - init_state.position))
                   * init_state.rotation).vec
        rel_rot = state.rotation * init_state.rotation.conj()

        tf = TransformStamped()
        tf.header.stamp = rospy.Time.now()
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_footprint"
        tf.transform.translation.x = -rel_pos[2]
        tf.transform.translation.y = -rel_pos[0]
        tf.transform.translation.z = rel_pos[1]
        tf.transform.rotation.x = -rel_rot.z
        tf.transform.rotation.y = -rel_rot.x
        tf.transform.rotation.z = rel_rot.y
        tf.transform.rotation.w = rel_rot.w
        self._tf_brdcast.sendTransform(tf)

    def _publish_scan(self):
        self._scan_msg.header.stamp = rospy.Time.now()
        scan_d = np.concatenate((self._last_obs["bl_scan"],
                                 self._last_obs["fl_scan"],
                                 self._last_obs["fr_scan"],
                                 self._last_obs["br_scan"]), 1)
        self._scan_msg.ranges = (scan_d[0, ::-1] * self._scan_d_to_rho).tolist()
        self._scan_pub.publish(self._scan_msg)

    def _publish_rgbd(self):
        rgb = self._last_obs["rgb"][:, :, :3]
        rgb_msg = self._cv_bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        self._rgb_pub.publish(rgb_msg)

        depth = (self._last_obs["depth"] * 1000).astype(np.uint16)
        depth_msg = self._cv_bridge.cv2_to_imgmsg(depth, encoding="mono16")
        self._depth_pub.publish(depth_msg)

    def _step_sim(self):
        if not self._has_cmd:
            return

        now = rospy.Time.now()
        if now - self._last_cmd_vel > self._cmd_timeout:
            rospy.logwarn("Last velocity command timed out, cancelling it.")
            self._vel_ctrl.linear_velocity.z = 0
            self._vel_ctrl.angular_velocity.y = 0
            self._has_cmd = False

        s = self._sim.get_rigid_state(self._agent_obj_id)
        dt = self._rate.sleep_dur.to_sec()
        nxt_s = self._vel_ctrl.integrate_transform(dt, s)
        nxt_pos = self._sim.step_filter(s.translation, nxt_s.translation)
        self._sim.set_translation(nxt_pos, self._agent_obj_id)
        self._sim.set_rotation(nxt_s.rotation, self._agent_obj_id)
        self._last_obs = self._sim.get_sensor_observations()


if __name__ == "__main__":
    sim_node = HabitatSimNode()
    sim_node.loop()
