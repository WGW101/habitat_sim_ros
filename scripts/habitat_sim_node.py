#!/usr/bin/env python3
#-*-coding: utf8-*-
import rospy
from cv_bridge import CvBridge
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from std_srvs.srv import Empty, EmptyResponse

import numpy as np
import quaternion
import habitat_sim


class HabitatSimNode:
    def __init__(self):
        rospy.init_node("habitat_sim")

        self._seed = rospy.get_param("~sim/seed", None)
        self._rng = np.random.default_rng(self._seed)

        self._sensor_specs = []
        self._static_tfs = []
        self._init_scan()
        self._init_rgbd()
        self._init_odom()
        self._init_sim()

        rospy.Service("~reset", Empty, self.reset)
        self._needs_reset = False


    def reset(self, req):
        self._needs_reset = True
        if self._use_sim_time:
            self._clock_msg = Clock()
        return EmptyResponse()


    def loop(self):
        self._broadcast_static_tf()
        while not rospy.is_shutdown():
            if self._needs_reset:
                self._last_obs = self._sim.reset()
                self._needs_reset = False
            self._broadcast_tf()
            self._publish_scan()
            self._publish_rgbd()
            self._step_sim()
            if self._use_sim_time:
                self._clock_msg.clock += self._rate.sleep_dur
                self._clock_pub.publish(self._clock_msg)
            else:
                self._rate.sleep()

    def _init_scan(self):
        rospy.loginfo("Setting up scan sensor")
        x = rospy.get_param("~scan/position/x", 0.0)
        y = rospy.get_param("~scan/position/y", 0.0)
        z = rospy.get_param("~scan/position/z", 0.45)
        r_min = rospy.get_param("~scan/range/min", 0.0)
        r_max = rospy.get_param("~scan/range/max", 10.0)
        n_rays = rospy.get_param("~scan/num_rays", 360)

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
        tf.transform.translation.y = y
        tf.transform.translation.z = z
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
        rectif = np.sqrt(np.linspace(-1, 1, n_rays_per_scan)**2 + 1)
        self._scan_rect = np.tile(rectif, (4,))[::-1]
        self._scan_bias = rospy.get_param("~scan/noise/bias", 0.0)
        self._scan_stdev = rospy.get_param("~scan/noise/stdev", 0.0)

        self._scan_pub = rospy.Publisher("~scan", LaserScan, queue_size=1)

    def _init_rgbd(self):
        rospy.loginfo("Setting up rgbd sensor")
        q = np.quaternion(0.5, -0.5, 0.5, -0.5)
        for sensor, sensor_type in (("rgb", habitat_sim.sensor.SensorType.COLOR),
                                    ("depth", habitat_sim.sensor.SensorType.DEPTH)):
            x = rospy.get_param(f"~{sensor}/position/x", 0.0)
            y = rospy.get_param(f"~{sensor}/position/y", 0.0)
            z = rospy.get_param(f"~{sensor}/position/z", 0.6)
            tilt = np.radians(rospy.get_param(f"~{sensor}/orientation/tilt", 0.0))
            h = rospy.get_param(f"~{sensor}/height", 240)
            w = rospy.get_param(f"~{sensor}/width", 320)
            hfov = rospy.get_param(f"~{sensor}/hfov", 60)
            f = 0.5 * w / np.tan(0.5 * hfov)

            info_pub = rospy.Publisher(f"~camera/{sensor}/camera_info", CameraInfo,
                                       queue_size=1, latch=True)
            cam_info = CameraInfo()
            cam_info.header.stamp = rospy.Time.now()
            cam_info.header.frame_id = f"camera/{sensor}/optical_frame"
            cam_info.height = h
            cam_info.width = w
            cam_info.distortion_model = "plumb_bob"
            cam_info.D = [0, 0, 0, 0, 0]
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
            tf.transform.translation.y = y
            tf.transform.translation.z = z
            q_tilt = np.quaternion(np.cos(0.5 * tilt), 0, np.sin(0.5*tilt), 0)
            rot = q_tilt * q
            tf.transform.rotation.x = rot.x
            tf.transform.rotation.y = rot.y
            tf.transform.rotation.z = rot.z
            tf.transform.rotation.w = rot.w
            self._static_tfs.append(tf)

            spec = habitat_sim.sensor.SensorSpec()
            spec.uuid = sensor
            spec.sensor_type = sensor_type
            spec.position = [-y, z, -x]
            spec.orientation = [-tilt, 0, 0]
            spec.resolution = [h, w]
            spec.parameters["hfov"] = str(hfov)
            self._sensor_specs.append(spec)

        self._rgb_pub = rospy.Publisher("~camera/rgb/image_raw", Image, queue_size=1)
        self._depth_pub = rospy.Publisher("~camera/depth/image_raw", Image, queue_size=1)
        self._cv_bridge = CvBridge()

    def _init_odom(self):
        self._odom_lin_bias = rospy.get_param("~odom/linear/noise/bias", 0.0)
        self._odom_lin_stdev = rospy.get_param("~odom/linear/noise/stdev", 0.0)
        self._odom_ang_bias = rospy.get_param("~odom/angular/noise/bias", 0.0)
        self._odom_ang_stdev = rospy.get_param("~odom/angular/noise/stdev", 0.0)
        self._odom_pos_drift = np.array([0.0, 0.0, 0.0])
        self._odom_rot_drift = np.quaternion(1.0, 0.0, 0.0, 0.0)
        self._tf_brdcast = TransformBroadcaster()

    def _init_sim(self):
        rospy.loginfo("Setting up simulator")
        self._use_sim_time = rospy.get_param("/use_sim_time", False)
        if self._use_sim_time:
            self._clock_msg = Clock()
            self._clock_pub = rospy.Publisher("/clock", Clock, queue_size=1)
        self._rate = rospy.Rate(rospy.get_param("~sim/rate", 60.0))

        sim_cfg = habitat_sim.sim.SimulatorConfiguration()
        sim_cfg.allow_sliding = rospy.get_param("~sim/allow_sliding", True)
        sim_cfg.scene_id = rospy.get_param("~sim/scene_path") # required!
        sim_cfg.random_seed = self._seed

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = rospy.get_param("~agent/height", 0.8)
        agent_cfg.radius = rospy.get_param("~agent/radius", 0.2)
        agent_cfg.sensor_specifications = self._sensor_specs

        self._vel_ctrl = habitat_sim.physics.VelocityControl()
        self._vel_ctrl.controlling_lin_vel = True
        self._vel_ctrl.lin_vel_is_local = True
        self._vel_ctrl.controlling_ang_vel = True
        self._vel_ctrl.ang_vel_is_local = True

        cfg = habitat_sim.simulator.Configuration(sim_cfg, [agent_cfg])
        self._sim = habitat_sim.simulator.Simulator(cfg)
        mngr = self._sim.get_object_template_manager()
        tmpl_id = mngr.get_template_ID_by_handle(*mngr.get_template_handles('cylinderSolid'))
        self._agent_obj_id = self._sim.add_object(tmpl_id, self._sim.get_agent(0).scene_node)

        state = habitat_sim.agent.AgentState()
        if rospy.get_param("~agent/start_position/random", True):
            state.position = self._sim.pathfinder.get_random_navigable_point()
        else:
            state.position[0] = -rospy.get_param("~agent/start_position/y", 0.0)
            state.position[1] = rospy.get_param("~agent/start_position/z", 0.0)
            state.position[2] = -rospy.get_param("~agent/start_position/x", 0.0)
        if rospy.get_param("~agent/start_orientation/random", True):
            yaw = 2 * np.pi * np.random.random()
        else:
            yaw = np.radians(rospy.get_param("~agent/start_orientation/yaw", 0.0))
        state.rotation = np.quaternion(np.cos(0.5 * yaw), 0, np.sin(0.5 * yaw), 0)

        self._sim.get_agent(0).set_state(state, is_initial=True)
        self._last_obs = self._sim.get_sensor_observations()

        self._cmd_vel_sub = rospy.Subscriber("~cmd_vel", Twist, self._on_cmd_vel)
        self._last_cmd_vel_time = rospy.Time.now()
        self._cmd_timeout = rospy.Duration(rospy.get_param("~agent/ctrl/timeout", 1.0))
        self._has_cmd = False

    def _on_cmd_vel(self, msg):
        self._vel_ctrl.linear_velocity.z = -msg.linear.x
        self._vel_ctrl.angular_velocity.y = msg.angular.z
        self._last_cmd_vel_time = rospy.Time.now()
        self._has_cmd = True

    def _broadcast_static_tf(self):
        static_tf_brdcast = StaticTransformBroadcaster()
        static_tf_brdcast.sendTransform(self._static_tfs)

    def _broadcast_tf(self):
        state = self._sim.get_agent(0).get_state()
        init_state = self._sim.get_agent(0).initial_state

        if self._has_cmd: # Only drift when moving...
            if self._odom_ang_stdev > 0:
                drift = self._rng.normal(self._odom_ang_bias, self._odom_ang_stdev)
                self._odom_rot_drift *= np.quaternion(np.cos(0.5 * drift), 0,
                                                      np.sin(0.5 * drift), 0)
                state.rotation *= self._odom_rot_drift

            if self._odom_lin_stdev > 0:
                drift = self._rng.normal(self._odom_lin_bias, self._odom_lin_stdev)
                heading = (state.rotation
                           * np.quaternion(0, 0, 0, -1)
                           * state.rotation.conj()).vec
                self._odom_pos_drift += drift * heading
                state.position += self._odom_pos_drift

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
        scan_r = scan_d[0, ::-1] * self._scan_rect
        if self._scan_stdev > 0:
            scan_r += self._rng.normal(self._scan_bias, self._scan_stdev, scan_r.shape)
        self._scan_msg.ranges = scan_r.tolist()
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
        if now - self._last_cmd_vel_time > self._cmd_timeout:
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
