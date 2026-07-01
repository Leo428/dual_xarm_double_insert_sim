from pathlib import Path
import queue
import threading
from typing import Any, Literal

import cv2
import gymnasium as gym
from gymnasium import spaces
from loop_rate_limiters import RateLimiter
import mink
import mujoco
from mujoco import Renderer
import numpy as np
from scipy.spatial.transform import Rotation as R

from dual_xarms_sim.mujoco_gym_env import GymRenderingSpec
from dual_xarms_sim.mujoco_gym_env import MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "ufactory_xarm7" / "insert_scene.xml"

LEFT_HOME = np.asarray([-0.35, 0.4, 0.2, 0, 0.7071068, -0.7071068, 0])
RIGHT_HOME = np.asarray([0.35, 0.4, 0.2, 0, 0.7071068, -0.7071068, 0])
LEFT_CARTESIAN_BOUNDS = np.asarray([[-0.7, 0.2, 0], [0.1, 0.6, 0.3]])
# LEFT_EULER_BOUNDS = np.asarray([[-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]])
RIGHT_CARTESIAN_BOUNDS = np.asarray([[-0.1, 0.2, 0], [0.7, 0.6, 0.3]])
# RIGHT_EULER_BOUNDS = np.asarray([[-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]])
_PEG_SAMPLING_BOUNDS = np.asarray([[0 - 0.075, 0.3 - 0.075], [0 + 0.075, 0.3 + 0.075]]) # 15cm x 15cm
_LEFT_SAMPLING_BOUNDS = np.asarray([[-0.2 - 0.075, 0.3 - 0.075], [-0.2 + 0.075, 0.3 + 0.075]]) # 15cm x 15cm
_RIGHT_SAMPLING_BOUNDS = np.asarray([[0.2 - 0.075, 0.3 - 0.075], [0.2 + 0.075, 0.3 + 0.075]]) # 15cm x 15cm

# Define joint names based on the xarm7 structure from your model
_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]
# All joints on xarm7 are assumed to have similar velocity limits
_VELOCITY_LIMITS = dict.fromkeys(_JOINT_NAMES, np.pi)
_HOME_JOINT_QPOS = np.array([0, -0.25891, -0.00020, 1.03223, 0, 1.31830, 0, 0, -0.25891, -0.00020, 1.03223, 0, 1.31830, 0])
_HOME_JOINT_CTRL = np.array([0.785398163, -0.247, 0, 0.909, 0, 1.15644, 0, 0])
_MAX_LINEAR_VELOCITY = 1 # m/s
_MAX_ANGULAR_VELOCITY = np.pi/3 # rad/s

class ImageDisplayer(threading.Thread):
    def __init__(self, queue: queue.Queue, heatmap_path: str = None, display_size=(672, 672)):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.heatmap = None
        self.expanded_alpha = None
        self.display_size = display_size  # Size to display the image (width, height)
        self.window_initialized = False
        self.show_overlay = False  # Start with overlay hidden

        if heatmap_path:
            try:
                # Load heatmap and convert to BGR immediately
                heatmap = cv2.imread(heatmap_path, cv2.IMREAD_UNCHANGED)
                if heatmap is None:
                    print(f"Failed to load heatmap from {heatmap_path}")
                    return

                # Process alpha channel if present
                if heatmap.shape[2] == 4:
                    self.heatmap_bgr = heatmap[:, :, :3]  # BGR format from OpenCV
                    # Pre-expand alpha for faster blending
                    self.expanded_alpha = np.expand_dims(heatmap[:, :, 3] / 255.0, axis=2)
                else:
                    self.heatmap_bgr = heatmap  # Already in BGR
                    self.expanded_alpha = np.ones((heatmap.shape[0], heatmap.shape[1], 1))

                print(f"Loaded heatmap with shape {self.heatmap_bgr.shape}")
            except Exception as e:
                print(f"Error loading heatmap: {e}")

    def set_overlay_visible(self, visible):
        """Toggle visibility of the heatmap overlay"""
        self.show_overlay = visible

    def run(self):
        # Create a resizable window
        cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
        while True:
            try:
                name, rgb = self.queue.get()
                # Convert RGB to BGR for OpenCV
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Apply heatmap overlay if available AND visibility is enabled
                if self.expanded_alpha is not None and self.show_overlay:
                    # Fast alpha blending with pre-calculated alpha
                    bgr = (bgr * (1 - self.expanded_alpha) +
                           self.heatmap_bgr * self.expanded_alpha).astype(np.uint8)

                # Resize for display (maintain aspect ratio)
                display_img = cv2.resize(bgr, self.display_size)

                # If first run, set window size
                if not self.window_initialized:
                    cv2.resizeWindow("Camera View", self.display_size[0], self.display_size[1])
                    self.window_initialized = True

                cv2.imshow("Camera View", display_img)
                cv2.waitKey(1)

            except Exception as e:
                print(f"Error in ImageDisplayer: {e}")
                continue

class DoubleInsertDualXarmsGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        seed: int = 0,
        control_freq: int = 60, # 60 Hz
        time_limit: int = 2 * 60, # in seconds
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(height=224, width=224),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
        run_ik: bool = True,
        overlay_heatmap: str = None,
        rotation_range: tuple[float, float] = (-np.pi/4, np.pi/4),  # OOD: can be (-np.pi/3, np.pi/3)
        color_randomization: bool = False,  # OOD: enable color randomization
    ):
        self.control_freq = control_freq
        self.step_counter = 0
        self.MAX_STEPS = time_limit * control_freq
        self.MAX_LINEAR_VELOCITY = _MAX_LINEAR_VELOCITY / control_freq
        self.MAX_ANGULAR_VELOCITY = _MAX_ANGULAR_VELOCITY / control_freq
        self.rotation_range = rotation_range
        self.color_randomization = color_randomization

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=1 / control_freq,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(control_freq),
        }

        self.render_mode = render_mode
        self.camera_names = ["left/top", "left/wrist", "right/top", "right/wrist"]
        self.image_obs = image_obs
        self._viewer = None

                 # Initialize display for overlay mode
        self._use_overlay = render_mode == "human" and overlay_heatmap is not None
        self._frames_queue = None
        self._displayer = None

        if render_mode == "human":
            if overlay_heatmap is not None:
                self._frames_queue = queue.Queue(maxsize=10)
                # Use a larger display size (3x the original size)
                self._displayer = ImageDisplayer(
                    self._frames_queue,
                    overlay_heatmap,
                    display_size=(224 * 3, 224 * 3)
                )
                self._displayer.start()
            else:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=True, show_right_ui=True)

        # Initialize renderer for all render modes
        self._renderer = Renderer(self.model, width=render_spec.width, height=render_spec.height)

        joint_names = []
        self.velocity_limits = {}
        for prefix in ["left", "right"]:
            for n in _JOINT_NAMES:
                name = f"{prefix}/{n}"
                joint_names.append(name)
                self.velocity_limits[name] = _VELOCITY_LIMITS[n]
        self.arm_dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.arm_actuator_ids = np.array([self.model.actuator(name).id for name in joint_names])

        gripper_names = [f"{prefix}/gripper" for prefix in ["left", "right"]]
        self._gripper_ctrl_ids = [self._model.actuator(gripper_name).id for gripper_name in gripper_names]
        self._tcp_site_ids = [self._model.site(f"{side}/link_tcp").id for side in ["left", "right"]]
        # self._block_z = self._model.geom("block").size[2]

        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict(
                {
                    "left/tcp_pose": spaces.Box( # world frame, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "left/tcp_vel": spaces.Box( # world frame, linear + angular euler
                        -np.inf, np.inf, shape=(6,), dtype=np.float32
                    ),
                    "left/ego_tcp_pose": spaces.Box( # head camera frame, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "left/ego_tcp_vel": spaces.Box( # head camera frame, linear + angular euler
                        -np.inf, np.inf, shape=(6,), dtype=np.float32
                    ),
                    "left/relative2_tcp_pose": spaces.Box( # relative to right tcp, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "left/relative2_tcp_vel": spaces.Box( # relative to right tcp, linear + angular euler
                        -np.inf, np.inf, shape=(6,), dtype=np.float32
                    ),
                    "left/gripper_pos": spaces.Box(
                        -np.inf, np.inf, shape=(1,), dtype=np.float32
                    ),
                    "left/joint_qpos": spaces.Box(
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "right/tcp_pose": spaces.Box( # world frame, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "right/tcp_vel": spaces.Box( # world frame
                        -np.inf, np.inf, shape=(6,), dtype=np.float32 # linear + angular euler
                    ),
                    "right/ego_tcp_pose": spaces.Box( # head camera frame, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "right/ego_tcp_vel": spaces.Box( # head camera frame, linear + angular euler
                        -np.inf, np.inf, shape=(6,), dtype=np.float32
                    ),
                    "right/relative2_tcp_pose": spaces.Box( # relative to left tcp, pos + quat
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                    "right/relative2_tcp_vel": spaces.Box( # relative to left tcp, linear + angular euler
                        -np.inf, np.inf, shape=(6,), dtype=np.float32
                    ),
                    "right/gripper_pos": spaces.Box(
                        -np.inf, np.inf, shape=(1,), dtype=np.float32
                    ),
                    "right/joint_qpos": spaces.Box(
                        -np.inf, np.inf, shape=(7,), dtype=np.float32
                    ),
                }
            ),
            "images": gym.spaces.Dict(
                {
                    "left/top": gym.spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
                    "left/wrist": gym.spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
                    "right/top": gym.spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
                    "right/wrist": gym.spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
                }
            )
        })

        # left tcp pos delta, left tcp euler delta, left gripper pos,
        # right tcp pos delta, right tcp euler delta, right gripper pos
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        self.ik_configuration = mink.Configuration(self.model)
        # Task definitions using mink library
        self.l_ee_task = mink.FrameTask(
            frame_name="left/link_tcp",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.r_ee_task = mink.FrameTask(
            frame_name="right/link_tcp",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(self.model, cost=1e-4)
        self.tasks = [self.l_ee_task, self.r_ee_task, self.posture_task]
        # Fetch geometry IDs for collision avoidance
        l_wrist_geoms = mink.get_subtree_geom_ids(self.model, self.model.body("left/link7").id)
        r_wrist_geoms = mink.get_subtree_geom_ids(self.model, self.model.body("right/link7").id)
        l_upper_arm_geoms = mink.get_subtree_geom_ids(self.model, self.model.body("left/link1").id)
        r_upper_arm_geoms = mink.get_subtree_geom_ids(self.model, self.model.body("right/link1").id)

        # Define geometry IDs for the environment if needed (example: table or frames)
        # You would need to define these based on your actual environment setup
        table_geoms = ["floor"]  # Placeholder, replace with actual geom ID(s)

        # Define collision pairs
        collision_pairs = [
            # (l_wrist_geoms, r_wrist_geoms),  # Avoid collisions between the left and right end-effectors
            (l_upper_arm_geoms + r_upper_arm_geoms, table_geoms),  # Avoid collisions between arms and the table
            (l_upper_arm_geoms, r_upper_arm_geoms),  # Avoid collisions between the left and right arms
        ]
        collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=collision_pairs,  # type: ignore
            minimum_distance_from_collisions=0.01,
            collision_detection_distance=0.05,
        )
        self.ik_limits = [
            mink.ConfigurationLimit(model=self.model),
            mink.VelocityLimit(self.model, self.velocity_limits),
            collision_avoidance_limit,
        ]
        self.ik_pos_threshold = 1e-2
        self.ik_ori_threshold = 1e-2
        self.ik_solver = "quadprog"
        self.ik_rate = RateLimiter(200, name="IK Rate")  # 200 Hz

    def set_ik_targets(self, l_pos, l_quat, r_pos, r_quat, steps=5):
        self.data.mocap_pos[0], self.data.mocap_quat[0] = l_pos, l_quat
        self.data.mocap_pos[1], self.data.mocap_quat[1] = r_pos, r_quat
        self.l_ee_task.set_target(mink.SE3.from_mocap_name(self.model, self.data, "left/target"))
        self.r_ee_task.set_target(mink.SE3.from_mocap_name(self.model, self.data, "right/target"))

        for ik_step in range(steps):
            for ik_iter in range(2):
                try:
                    vel = mink.solve_ik(
                        self.ik_configuration,
                        self.tasks,
                        self.control_dt,
                        self.ik_solver,
                        limits=self.ik_limits,
                        damping=1e-5,
                    )
                    self.ik_configuration.integrate_inplace(vel, self.ik_rate.dt)
                    l_err = self.l_ee_task.compute_error(self.ik_configuration)
                    r_err = self.r_ee_task.compute_error(self.ik_configuration)
                    if np.linalg.norm(l_err[:3]) <= self.ik_pos_threshold and np.linalg.norm(l_err[3:]) <= self.ik_ori_threshold \
                        and np.linalg.norm(r_err[:3]) <= self.ik_pos_threshold and np.linalg.norm(r_err[3:]) <= self.ik_ori_threshold:
                        break

                except Exception:
                    # print(f"IK error: {e}")
                    pass

            self.data.ctrl[self.arm_actuator_ids] = self.ik_configuration.q[self.arm_dof_ids]
            mujoco.mj_step(self.model, self.data)

            # Update viewers/displays
            if self._use_overlay and self._frames_queue is not None:
                self._update_overlay()
            elif self._viewer and self._viewer.is_running():
                self._viewer.sync()

    def _update_overlay(self):
        """Update the overlay display with the current frame"""
        try:
            frames = self.render()
            # Find the right/top camera frame
            for i, cam_name in enumerate(self.camera_names):
                if cam_name == "right/top":
                    self._frames_queue.put(("right/top", frames[i]), block=False)
                    break
        except queue.Full:
            pass  # Skip frame if queue is full

    def reset(
        self, seed=None, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment."""
        self.step_counter = 0
        super().reset(seed=seed, **kwargs)
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        self._data.qpos[self.arm_dof_ids] = _HOME_JOINT_QPOS
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        self._data.mocap_pos[0], self._data.mocap_quat[0] = LEFT_HOME[:3], LEFT_HOME[3:]
        self._data.mocap_pos[1], self._data.mocap_quat[1] = RIGHT_HOME[:3], RIGHT_HOME[3:]
        mujoco.mj_forward(self._model, self._data)

        # Sample a new peg position.
        peg_xy = np.random.uniform(*_PEG_SAMPLING_BOUNDS)
        peg_rot = np.array([0, 0, np.random.uniform(*self.rotation_range)])
        self._data.jnt("peg").qpos[:3] = (*peg_xy, 0.1)
        self._data.jnt("peg").qpos[3:] = R.from_euler("xyz", peg_rot).as_quat(scalar_first=True)
        # Sample a new peg position.
        left_socket_xy = np.random.uniform(*_LEFT_SAMPLING_BOUNDS)
        left_socket_rot = np.array([0, 0, np.random.uniform(*self.rotation_range)])
        self._data.jnt("left/socket/joint").qpos[:3] = (*left_socket_xy, 0.1)
        self._data.jnt("left/socket/joint").qpos[3:] = R.from_euler("xyz", left_socket_rot).as_quat(scalar_first=True)
        # Sample a new peg position.
        right_socket_xy = np.random.uniform(*_RIGHT_SAMPLING_BOUNDS)
        right_socket_rot = np.array([0, 0, np.random.uniform(*self.rotation_range)])
        self._data.jnt("right/socket/joint").qpos[:3] = (*right_socket_xy, 0.1)
        self._data.jnt("right/socket/joint").qpos[3:] = R.from_euler("xyz", right_socket_rot).as_quat(scalar_first=True)

        # Apply color randomization if enabled (OOD Experiment 2)
        if self.color_randomization:
            self._randomize_colors()

        mujoco.mj_forward(self._model, self._data)

        self.ik_configuration.update(self._data.qpos)
        self.posture_task.set_target_from_configuration(self.ik_configuration)
        self.set_ik_targets(
            LEFT_HOME[:3], LEFT_HOME[3:], RIGHT_HOME[:3], RIGHT_HOME[3:]
        )

        # Update the overlay if using it
        if self._use_overlay:
            self._update_overlay()

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray
            dimensions are: left tcp pos delta, left tcp euler delta, left gripper pos,
                right tcp pos delta, right tcp euler delta, right gripper pos

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        left_tcp_pos_delta = self.limit_offset_norm(action[:3], self.MAX_LINEAR_VELOCITY)
        left_tcp_euler_delta = self.limit_offset_norm(action[3:6], self.MAX_ANGULAR_VELOCITY)
        right_tcp_pos_delta = self.limit_offset_norm(action[7:10], self.MAX_LINEAR_VELOCITY)
        right_tcp_euler_delta = self.limit_offset_norm(action[10:13], self.MAX_ANGULAR_VELOCITY)

        # # Set the mocap position.
        left_pos = self._data.mocap_pos[0].copy()
        left_npos = np.clip(left_pos + left_tcp_pos_delta, *LEFT_CARTESIAN_BOUNDS)
        left_quat = self._data.mocap_quat[0].copy()
        left_dquat = R.from_euler("xyz", left_tcp_euler_delta)
        left_nquat = (left_dquat * R.from_quat(left_quat, scalar_first=True)).as_quat(scalar_first=True)

        right_pos = self._data.mocap_pos[1].copy()
        right_npos = np.clip(right_pos + right_tcp_pos_delta, *RIGHT_CARTESIAN_BOUNDS)
        right_quat = self._data.mocap_quat[1].copy()
        right_dquat = R.from_euler("xyz", right_tcp_euler_delta)
        right_nquat = (right_dquat * R.from_quat(right_quat, scalar_first=True)).as_quat(scalar_first=True)

        self.set_ik_targets(left_npos, left_nquat, right_npos, right_nquat)

        # Set gripper grasp.
        left_g = self._data.ctrl[self._gripper_ctrl_ids[0]] / 255
        left_dg = action[6] * 0.1
        left_ng = np.clip(left_g + left_dg, 0.0, 1.0)
        right_g = self._data.ctrl[self._gripper_ctrl_ids[1]] / 255
        right_dg = action[13] * 0.1
        right_ng = np.clip(right_g + right_dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_ids[0]] = left_ng * 255
        self._data.ctrl[self._gripper_ctrl_ids[1]] = right_ng * 255

        obs = self._compute_observation()
        rew = self._compute_reward()
        # terminated = self.time_limit_exceeded()
        done = True if rew == 3.0 else False
        self.step_counter += 1
        truncated = self.step_counter >= self.MAX_STEPS

        # Update the overlay display if needed
        if self._use_overlay:
            self._update_overlay()

        return obs, rew, done, truncated, {}

    # directly takes in joint angles from both arms and grippers, (16,)
    def step_joints(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        left_gripper = action[7]
        right_gripper = action[15]
        for _ in range(10):
            self.data.ctrl[self.arm_actuator_ids] = np.concatenate((action[:7], action[8:15]))
            self.data.ctrl[self._gripper_ctrl_ids[0]] = left_gripper * 255
            self.data.ctrl[self._gripper_ctrl_ids[1]] = right_gripper * 255
            mujoco.mj_step(self.model, self.data)

        obs = self._compute_observation()
        rew = self._compute_reward()
        done = True if rew == 4.0 else False

        # Update viewer or overlay
        if self._use_overlay:
            self._update_overlay()
        elif self._viewer and self._viewer.is_running():
            self._viewer.sync()

        return obs, rew, done, False, {}

    def render(self):
        rendered_frames = []
        for cam_name in self.camera_names:
            self._renderer.update_scene(self.data, camera=cam_name)
            rendered_frames.append(self._renderer.render())

        # If in overlay mode, update the display
        if self._use_overlay and self._frames_queue is not None:
            try:
                for i, cam_name in enumerate(self.camera_names):
                    if cam_name == "right/top":
                        self._frames_queue.put(("right/top", rendered_frames[i]), block=False)
                        break
            except queue.Full:
                pass  # Skip frame if queue is full

        return rendered_frames

    def _compute_observation(self) -> dict:
        # IMPORTANT NOTE:
        # in observation, the quat from mujoco is scalar first, but we should keep it scalar last
        obs = {}
        obs["state"] = {}

        for side in ["left", "right"]:
            # in world frame
            tcp_pos = self._data.sensor(f"{side}/tcp_pos").data
            tcp_quat = np.roll(self._data.sensor(f"{side}/tcp_quat").data, -1)
            obs["state"][f"{side}/tcp_pose"] = np.concatenate([tcp_pos, tcp_quat]).astype(np.float32)
            tcp_vel = self._data.sensor(f"{side}/tcp_vel").data
            tcp_angvel = self._data.sensor(f"{side}/tcp_angvel").data
            obs["state"][f"{side}/tcp_vel"] = np.concatenate([tcp_vel, tcp_angvel]).astype(np.float32)

            # in head camera frame
            ego_tcp_pos = self._data.sensor(f"{side}/ego_tcp_pos").data
            ego_tcp_quat = np.roll(self._data.sensor(f"{side}/ego_tcp_quat").data, -1)
            obs["state"][f"{side}/ego_tcp_pose"] = np.concatenate([ego_tcp_pos, ego_tcp_quat]).astype(np.float32)
            ego_tcp_vel = self._data.sensor(f"{side}/ego_tcp_vel").data
            ego_tcp_angvel = self._data.sensor(f"{side}/ego_tcp_angvel").data
            obs["state"][f"{side}/ego_tcp_vel"] = np.concatenate([ego_tcp_vel, ego_tcp_angvel]).astype(np.float32)

            # relative to the other side tcp
            wrt2other_tcp_pos = self._data.sensor(f"{side}/relative2_tcp_pos").data
            wrt2other_tcp_quat = np.roll(self._data.sensor(f"{side}/relative2_tcp_quat").data, -1)
            obs["state"][f"{side}/relative2_tcp_pose"] = np.concatenate([wrt2other_tcp_pos, wrt2other_tcp_quat]).astype(np.float32)
            wrt2other_tcp_vel = self._data.sensor(f"{side}/relative2_tcp_vel").data
            wrt2other_tcp_angvel = self._data.sensor(f"{side}/relative2_tcp_angvel").data
            obs["state"][f"{side}/relative2_tcp_vel"] = np.concatenate([wrt2other_tcp_vel, wrt2other_tcp_angvel]).astype(np.float32)

            # joint qpos
            joint_qpos = self._data.qpos[self.arm_dof_ids]
            obs["state"][f"{side}/joint_qpos"] = joint_qpos[:7] if side == "left" else joint_qpos[7:]

        # gripper pos
        obs["state"]["left/gripper_pos"] = np.array(
            (self._data.ctrl[self._gripper_ctrl_ids[0]] / 255,), dtype=np.float32)
        obs["state"]["right/gripper_pos"] = np.array(
            (self._data.ctrl[self._gripper_ctrl_ids[1]] / 255,), dtype=np.float32)

        obs["images"] = {}
        images = self.render()
        for cam_name in self.camera_names:
            obs["images"][cam_name] = images.pop(0)

        return obs

    # TODO: update rewards for sim insertion task
    def _compute_reward(self) -> float:
        # Check if the block is in contact with the gripper
        all_contact_pairs = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            contact_pair = (self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name)
            all_contact_pairs.append(contact_pair)

        left_peg_inserted, right_peg_inserted = False, False
        right_socket_on_block, left_socket_on_block = False, False
        for contact_pair in all_contact_pairs:
            if contact_pair == ("peg", "left/socket/pin"):
                left_peg_inserted = True
            if contact_pair == ("peg", "right/socket/pin"):
                right_peg_inserted = True
            if contact_pair == ("block", "left/socket/wall_1") or contact_pair == ("block", "left/socket/wall_2") or \
                    contact_pair == ("block", "left/socket/wall_3") or contact_pair == ("block", "left/socket/wall_4"):
                left_socket_on_block = True and self.data.body("left/socket").xpos[2] > 0.1
            if contact_pair == ("block", "right/socket/wall_1") or contact_pair == ("block", "right/socket/wall_2") or \
                    contact_pair == ("block", "right/socket/wall_3") or contact_pair == ("block", "right/socket/wall_4"):
                right_socket_on_block = True and self.data.body("right/socket").xpos[2] > 0.1

        everything_on_block = left_peg_inserted and right_peg_inserted and left_socket_on_block and right_socket_on_block
        everything_lifted = self.data.body("peg").xpos[2] > 0.1 and \
                            self.data.body("left/socket").xpos[2] > 0.1 and \
                            self.data.body("right/socket").xpos[2] > 0.1

        if everything_on_block: # if everything is on the block
            return 3.0
        if left_peg_inserted and right_peg_inserted and everything_lifted:
            # if both pegs are inserted and everything is lifted
            return 2.0
        if left_peg_inserted or right_peg_inserted: # if left or right peg is inserted
            return 1.0
        return 0.0

    def close(self):
        if self._viewer:
            self._viewer.close()
        if hasattr(self._renderer, "close"):
            self._renderer.close()
        super().close()

    def _randomize_colors(self):
        """Randomize colors of blocks/pegs/sockets for OOD evaluation."""
        # Get geom IDs for objects we want to randomize
        geom_names_to_randomize = [
            "peg",
            "left/socket/wall_1", "left/socket/wall_2", "left/socket/wall_3", "left/socket/wall_4",
            "right/socket/wall_1", "right/socket/wall_2", "right/socket/wall_3", "right/socket/wall_4",
        ]

        # Randomize colors: sample RGB values in [0.2, 0.9] range to avoid too dark/bright
        for geom_name in geom_names_to_randomize:
            try:
                geom_id = self.model.geom(geom_name).id
                # Random RGB values (keep alpha=1.0)
                new_rgba = np.array([
                    np.random.uniform(0.2, 0.9),
                    np.random.uniform(0.2, 0.9),
                    np.random.uniform(0.2, 0.9),
                    1.0
                ], dtype=np.float32)
                self.model.geom(geom_name).rgba[:] = new_rgba
            except Exception:
                # Skip if geom doesn't exist or can't be modified
                pass

        # Handle block separately - it uses a material, so we need to override the geom's rgba
        # Setting rgba on the geom should override the material's rgba in MuJoCo
        try:
            block_rgba = np.array([
                np.random.uniform(0.2, 0.9),
                np.random.uniform(0.2, 0.9),
                np.random.uniform(0.2, 0.9),
                1.0
            ], dtype=np.float32)
            self.model.geom("block").rgba[:] = block_rgba
        except Exception:
            # If direct access fails, try to modify the material
            try:
                # Modify the material's rgba (this affects all geoms using this material)
                mat_id = self.model.material("M_simpleWood1").id
                new_rgba = np.array([
                    np.random.uniform(0.2, 0.9),
                    np.random.uniform(0.2, 0.9),
                    np.random.uniform(0.2, 0.9),
                    1.0
                ], dtype=np.float32)
                self.model.material("M_simpleWood1").rgba[:] = new_rgba
            except Exception:
                pass

        # Also randomize pin colors (they're red by default)
        pin_names = ["left/socket/pin", "right/socket/pin"]
        for pin_name in pin_names:
            try:
                new_rgba = np.array([
                    np.random.uniform(0.3, 0.95),
                    np.random.uniform(0.3, 0.95),
                    np.random.uniform(0.3, 0.95),
                    1.0
                ], dtype=np.float32)
                self.model.geom(pin_name).rgba[:] = new_rgba
            except Exception:
                pass

    def limit_offset_norm(self, offset, max_offset):
        # scale offset such that the max norm of offset is max_offset
        norm = np.linalg.norm(offset)
        if norm > max_offset:
            offset = offset / norm * max_offset
        return offset

