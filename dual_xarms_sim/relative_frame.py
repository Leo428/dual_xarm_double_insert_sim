from copy import deepcopy

import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R

from dual_xarms_sim.utils.transformation import compute_relative_poses_batch
from dual_xarms_sim.utils.transformation import compute_relative_velocities_batch
from dual_xarms_sim.utils.transformation import construct_adjoint_matrix
from dual_xarms_sim.utils.transformation import construct_homogeneous_matrix


class RelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "left/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                "right/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ......
            }
        ),
        ......
    }, and at least 14 DoF action space with (l_dx, l_dy, l_dz, l_drx, l_dry, l_drz, l_dgrip, ...).
    By convention, the 7th and 14th dimension of the action space is used for the gripper.

    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.adjoint_matrix = {
            "left": np.zeros((6, 6)),
            "right": np.zeros((6, 6)),
        }

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = {
                "left": np.zeros((4, 4)),
                "right": np.zeros((4, 4)),
            }

        self.observation_space = deepcopy(env.observation_space)
        for side in ["left", "right"]:
            if self.include_relative_pose:
                # Add relative pose to observation space
                self.observation_space["state"][f"{side}/wrist_tcp_pose"] = spaces.Box(
                    -np.inf, np.inf, shape=(7,)
                )
            self.observation_space["state"][f"{side}/wrist_tcp_vel"] = spaces.Box(
                -np.inf, np.inf, shape=(6,)
            )
            self.observation_space["state"][f"{side}/og_action"] = spaces.Box(
                -1, 1, shape=(7,)
            )

    def step(self, action: np.ndarray):
        """
            action is assumed to be (x, y, z, rx, ry, rz, gripper)
            the base frame for actions is w.r.t to the wrists's coordinate frame
        """
        # Transform action from wrist's coordinate frame to robot base's coord frame
        transformed_action = self.transform_action(action)

        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["og_intervene_action"] = deepcopy(info["intervene_action"])
            info["intervene_action"] = self.transform_action_inv(
                info["intervene_action"]
            )

        # Update adjoint matrix, so that the new observation returned is updated accordingly
        self.adjoint_matrix = {
            "left": construct_adjoint_matrix(obs["state"]["left/tcp_pose"]),
            "right": construct_adjoint_matrix(obs["state"]["right/tcp_pose"]),
        }

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        if "og_intervene_action" in info:
            transformed_obs["state"]["left/og_action"] = info["og_intervene_action"][:7]
            transformed_obs["state"]["right/og_action"] = info["og_intervene_action"][7:]
        else:
            transformed_obs["state"]["left/og_action"] = transformed_action[:7]
            transformed_obs["state"]["right/og_action"] = transformed_action[7:]

        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for side in ["left", "right"]:
            obs["state"][f"{side}/wrist_tcp_pose"] = obs["state"][f"{side}/tcp_pose"]
            obs["state"][f"{side}/wrist_tcp_vel"] = obs["state"][f"{side}/tcp_vel"]

            # Update adjoint matrix
            self.adjoint_matrix[side] = construct_adjoint_matrix(
                obs["state"][f"{side}/wrist_tcp_pose"]
            )
            if self.include_relative_pose:
                # Update transformation matrix from the reset pose's relative frame to base frame
                self.T_r_o_inv[side] = np.linalg.inv(
                    construct_homogeneous_matrix(obs["state"][f"{side}/wrist_tcp_pose"])
                )
            obs["state"][f"{side}/og_action"] = np.zeros(7)

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix
        """
        for side in ["left", "right"]:
            adjoint_inv = np.linalg.inv(self.adjoint_matrix[side])
            obs["state"][f"{side}/wrist_tcp_vel"] = adjoint_inv @ obs["state"][f"{side}/tcp_vel"]

            if self.include_relative_pose:
                T_b_o = construct_homogeneous_matrix(obs["state"][f"{side}/tcp_pose"])
                T_b_r = self.T_r_o_inv[side] @ T_b_o

                # Reconstruct transformed tcp_pose vector
                p_b_r = T_b_r[:3, 3]
                theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
                obs["state"][f"{side}/wrist_tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the adjoint matrix
        """
        new_action = action.copy() # to avoid modifying the original action
        # left arm
        new_action[:6] = self.adjoint_matrix["left"] @ action[:6]
        # right arm
        new_action[7:13] = self.adjoint_matrix["right"] @ action[7:13]
        return new_action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix.
        """
        new_action = action.copy() # to avoid modifying the original action
        # left arm
        new_action[:6] = np.linalg.inv(self.adjoint_matrix["left"]) @ action[:6]
        # right arm
        new_action[7:13] = np.linalg.inv(self.adjoint_matrix["right"]) @ action[7:13]
        return new_action

class WristRelativeTo(gym.ObservationWrapper):
    """
    This wrapper transforms the observation to be expressed in the wrist frame relative to the base frame.
    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = deepcopy(env.observation_space)

    def observation(self, observation):
        # TODO: currently it's using ee_pose instead of tcp_pose, need to fix that after
        # re-training the model
        left_tcp_pose = observation["state"]["left/tcp_pose"][None, ...]
        right_tcp_pose = observation["state"]["right/tcp_pose"][None, ...]
        left_tcp_vel = observation["state"]["left/tcp_vel"][None, ...]
        right_tcp_vel = observation["state"]["right/tcp_vel"][None, ...]
        observation["state"]["left/relative2_tcp_pose"] = compute_relative_poses_batch(left_tcp_pose, right_tcp_pose)[0]
        observation["state"]["right/relative2_tcp_pose"] = compute_relative_poses_batch(right_tcp_pose, left_tcp_pose)[0]
        observation["state"]["left/relative2_tcp_vel"] = compute_relative_velocities_batch(left_tcp_vel, right_tcp_vel, left_tcp_pose, right_tcp_pose)[0]
        observation["state"]["right/relative2_tcp_vel"] = compute_relative_velocities_batch(right_tcp_vel, left_tcp_vel, right_tcp_pose, left_tcp_pose)[0]
        return observation
