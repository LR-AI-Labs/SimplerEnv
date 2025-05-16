from typing import Optional, Sequence, List
import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from transformers import AutoModel, AutoProcessor
from collections import deque
from PIL import Image
import torch
import cv2 as cv

from simpler_env.utils.action.action_ensemble import ActionEnsembler

from gr00t.model.policy import Gr00tPolicy, L1Gr00tPolicy # type: ignore
from gr00t.experiment.data_config import DATA_CONFIG_MAP # type: ignore

class Gr00tInference:
    def __init__(
        self,
        saved_model_path: str = "",
        policy_setup: str = "google_robot",
        image_size: list[int] = [256, 320],
        exec_horizon: int = 1,
        use_diffusion: bool = False,
        use_regression: bool = False,
        embodiment_tag: str = "new_embodiment"
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.policy_setup = policy_setup

        print(f"*** policy_setup: {policy_setup} ***")
        policy_cfg = dict(
            pretrained_checkpoint=saved_model_path,
            embodiment_tag=embodiment_tag,
            use_diffusion=use_diffusion,
            use_regression=use_regression,
            data_config=policy_setup,
        )
        self.policy = self.get_gr00t_policy(policy_cfg)

        self.image_size = image_size
        self.obs_horizon = 1
        self.obs_interval = 1
        self.pred_action_horizon = 16
        self.image_history = deque(maxlen=self.obs_horizon)
        # self.exec_horizon = exec_horizon

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        
        self.task = None
        self.task_description = None
        
    def get_gr00t_policy(self, cfg) -> torch.nn.Module:
        embodiment_tag = cfg['embodiment_tag']
        data_config = DATA_CONFIG_MAP[cfg['data_config']]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if cfg["use_diffusion"]:
            policy = Gr00tPolicy(
                model_path=cfg["pretrained_checkpoint"],
                embodiment_tag=embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                device=device
            )
        elif cfg["use_regression"]:
            policy = L1Gr00tPolicy(
                model_path=cfg["pretrained_checkpoint"],
                l1_model_path=cfg["pretrained_checkpoint"],
                embodiment_tag=embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                device=device
            )
        else:
            raise ValueError("Unsupported model type. Please specify either diffusion or regression.")
        return policy

    def reset(self, task_description: str) -> None:
        self.image_history.clear()
        self.task_description = task_description
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)
        
        if image.shape[-1] == 4: # (H, W, 4) is RGBD
            image = image[:, :, :3] * 255 # remove the depth channel and scale to [0, 255]
        image = self._resize_image(image)
        self._add_image_to_history(image)
        
        assert "eef_pos" in kwargs
        eef_pos = kwargs["eef_pos"]
        if self.policy_setup == "google_robot":
            eef_pos = self.preprocess_googple_robot_proprio(eef_pos)
        else:
            raise NotImplementedError(f"Policy setup {self.policy_setup} not supported")
        
        inputs = {
            "video.image": image[np.newaxis, ...],
            "state.state": eef_pos[np.newaxis, ...],
            "action.action": np.zeros((self.pred_action_horizon, 7)),
            "annotation.human.task_description": [task_description]
        }

        # predict action (7-dof; un-normalize for bridgev2)
        raw_actions = self.policy.get_action(inputs)["action.action"]

        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(
                raw_actions[0, 6:7]
            ),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    def preprocess_googple_robot_proprio(self, eef_pos) -> np.array:
        """convert wxyz quat from simpler to xyzw used in fractal
        https://github.com/allenzren/open-pi-zero/blob/c3df7fb062175c16f69d7ca4ce042958ea238fb7/src/agent/env_adapter/simpler.py#L204
        """
        # StateEncoding.POS_QUAT: xyz + q_xyzw + gripper(closeness)
        quat_xyzw = np.roll(eef_pos[3:7], -1)
        gripper_width = eef_pos[
            7
        ]  # from simpler, 0 for close, 1 for open
        # need invert as the training data comes from closeness
        gripper_closedness = (
            1 - gripper_width
        )  # TODO(allenzren): change fractal data processing in training so also use gripper openness in proprio (as in bridge) instead of closedness
        raw_proprio = np.concatenate(
            (
                eef_pos[:3],
                quat_xyzw,
                [gripper_closedness],
            )
        )
        return raw_proprio
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA).astype(np.uint8)
        return image

    def _add_image_to_history(self, image: np.ndarray) -> None:
        if len(self.image_history) == 0:
            self.image_history.extend([image] * self.obs_horizon)
        else:
            self.image_history.append(image)

    def _obtain_image_history(self) -> List[Image.Image]:
        image_history = list(self.image_history)
        images = image_history[:: self.obs_interval]
        images = [Image.fromarray(image).convert("RGB") for image in images]
        return images

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate(
                    [a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1
                )
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(
                pred_actions[:, action_dim], label="predicted action"
            )
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
