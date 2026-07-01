import gymnasium as gym
import numpy as np

from dual_xarms_sim.utils.network import get_oculus_reading

class OculusIntervention(gym.ActionWrapper):
    def __init__(self, env, freq=10):
        super().__init__(env)
        self.oculus_freq = freq

        self.gripper_enabled = True
        if self.action_space.shape == (12,):
            self.gripper_enabled = False

    def action(self, action: np.ndarray):
        """
            Input:
            - action: policy action
            Output:
            - action: oculus action if button is pressed; else, policy action
            - intervened: True if oculus action is used; else, False
        """
        data = get_oculus_reading(timeout=1 / self.oculus_freq)
        if data is None:
            return action, {"intervened": False, "oculus_data": None}

        # if data["left_move_button"] or data["right_move_button"]:
        #     action[:3] = np.array([data["left_dx"], data["left_dy"], data["left_dz"]])
        #     # scale between [-1, 1]
        #     action[:3] = action[:3] / self.env.MAX_LINEAR_VELOCITY
        #     action[3:6] = np.array([data["left_drx"], data["left_dry"], data["left_drz"]])
        #     action[3:6] = action[3:6] / self.env.MAX_ANGULAR_VELOCITY
        #     action[6] = data["left_joystick"][0]
        #     action[7:10] = np.array([data["right_dx"], data["right_dy"], data["right_dz"]])
        #     action[7:10] = action[7:10] / self.env.MAX_LINEAR_VELOCITY
        #     action[10:13] = np.array([data["right_drx"], data["right_dry"], data["right_drz"]])
        #     action[10:13] = action[10:13] / self.env.MAX_ANGULAR_VELOCITY
        #     action[13] = data["right_joystick"][0]
        #     action = np.clip(action, -1, 1)
        #     return action, {"intervened": True, "oculus_data": data}
        if data["left_move_button"] or data["right_move_button"]:
            action[:3] = np.array([data["left_dx"], data["left_dy"], data["left_dz"]])
            action[:3] = action[:3]
            action[3:6] = np.array([data["left_drx"], data["left_dry"], data["left_drz"]])
            action[3:6] = action[3:6]
            action[6] = data["left_joystick"][0]
            action[7:10] = np.array([data["right_dx"], data["right_dy"], data["right_dz"]])
            action[7:10] = action[7:10]
            action[10:13] = np.array([data["right_drx"], data["right_dry"], data["right_drz"]])
            action[10:13] = action[10:13]
            action[13] = data["right_joystick"][0]
            return action, {"intervened": True, "oculus_data": data}

        return action, {"intervened": False, "oculus_data": data}

    def step(self, action):
        new_action, oculus_info = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if oculus_info["intervened"]:
            info["intervene_action"] = new_action
            info["left_a_button"] = oculus_info["oculus_data"]["left_a_button"]
            info["left_b_button"] = oculus_info["oculus_data"]["left_b_button"]
            info["right_a_button"] = oculus_info["oculus_data"]["right_a_button"]
            info["right_b_button"] = oculus_info["oculus_data"]["right_b_button"]
            if info["left_b_button"] or info["right_b_button"]:
                truncated = True

        return obs, rew, done, truncated, info
