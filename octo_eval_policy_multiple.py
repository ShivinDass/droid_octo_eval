import os
import time
import cv2
import copy
import numpy as np
import time
import imageio

from telemoma.utils.camera_utils import RealSenseCamera
from telemoma.human_interface.teleop_policy import TeleopPolicy
from telemoma.configs.only_spacemouse import teleop_config
from telemoma.robot_interface.franka.franka_gym import FrankaGym
from telemoma.utils.general_utils import AttrDict
from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import (
    HistoryWrapper,
    TemporalEnsembleWrapper,
    RHCWrapper,
    # UnnormalizeActionProprio,
)

import jax
import jax.numpy as jnp
from functools import partial

import gym

class EvalWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, teleop):
        super().__init__(env)

        self.teleop = teleop
        teleop.start()

    def process_obs(self, obs):
        processed_obs = {
            'proprio': obs['right'].astype(np.float32),
            'image_primary': np.array(cv2.resize(cv2.cvtColor(obs['primary_image'].astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256))).astype(float),
            'image_wrist': np.array(cv2.resize(cv2.cvtColor(obs['wrist_image'].astype(np.uint8), cv2.COLOR_RGB2BGR), (128, 128))).astype(float),
        }

        return processed_obs
    
    def step(self, action):
        teleop_action = self.teleop.get_action({})
        buttons = teleop_action.extra['buttons']
        if np.linalg.norm(teleop_action['right'][:6]) != 0:
            action[:] = teleop_action['right'][:7]

        # import time
        # time.sleep(0.1)
        # n_obs = self.process_obs(self.orig_obs)
        # print(action)
        n_obs, reward, done, info = self.env.step(AttrDict({'right': action}))
        n_obs = self.process_obs(n_obs)

        done = buttons.get('l_button', False)
        
        return n_obs, 0, done, False, {}
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.orig_obs = obs
        return self.process_obs(obs), {}

class UnnormalizeActionProprio(gym.Wrapper):
    """
    Un-normalizes the action and proprio.
    """

    def __init__(
        self,
        env: gym.Env,
        action_proprio_metadata: dict,
        normalization_type: str,
    ):
        self.action_proprio_metadata = jax.tree_map(
            lambda x: np.array(x),
            action_proprio_metadata,
            is_leaf=lambda x: isinstance(x, list),
        )
        self.normalization_type = normalization_type
        super().__init__(env)

    def unnormalize(self, data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        if self.normalization_type == "normal":
            return np.where(
                mask,
                (data * metadata["std"]) + metadata["mean"],
                data,
            )
        elif self.normalization_type == "bounds":
            return np.where(
                mask,
                ((data + 1) / 2 * (metadata["max"] - metadata["min"] + 1e-8))
                + metadata["min"],
                data,
            )
        else:
            raise ValueError(
                f"Unknown action/proprio normalization type: {self.normalization_type}"
            )

    def normalize(self, data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        if self.normalization_type == "normal":
            return np.where(
                mask,
                (data - metadata["mean"]) / (metadata["std"] + 1e-8),
                data,
            )
        elif self.normalization_type == "bounds":
            return np.where(
                mask,
                np.clip(
                    2
                    * (data - metadata["min"])
                    / (metadata["max"] - metadata["min"] + 1e-8)
                    - 1,
                    -1,
                    1,
                ),
                data,
            )
        else:
            raise ValueError(
                f"Unknown action/proprio normalization type: {self.normalization_type}"
            )

    def get_action(self, action):
        action = self.unnormalize(action, self.action_proprio_metadata["action"])
        return action

    def get_observation(self, obs):
        # return obs
        obs["proprio"] = self.normalize(
            obs["proprio"], self.action_proprio_metadata["proprio"]
        )
        return obs

    def step(self, action):
        action = self.get_action(action)
        obs, reward, done, trunc, info = self.env.step(action)
        obs = self.get_observation(obs)

        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)       
        obs = self.get_observation(obs)

        return obs, info

@jax.jit
def sample_actions(
    pretrained_model: OctoModel,
    observations,
    tasks,
    rng
):
    # add batch dim to observations
    observations = jax.tree_map(lambda x: x[None], observations)
    actions = pretrained_model.sample_actions(
        observations,
        tasks,
        rng=rng,
    )
    # remove batch dim
    return actions[0]

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped

def rollout_policy(policy_fn, env, teleop, save_vid=False):

    if save_vid:
        video = imageio.get_writer(f'experiments/videos/octo_{time.asctime().replace(" ", "-")}.mp4', fps=10)

    obs, info = env.reset(reset_arms=True)
    input("Press enter to start...")
    
    for i in range(1000):
        if i==0:
            obs, info = env.reset(reset_arms=False)
            continue
        policy_act = np.array(policy_fn(obs, task), dtype=np.float64)
        
        n_obs, reward, done, trunc, info = env.step(policy_act)

        if done:
            break
        if save_vid:
            img = obs['image_primary'][-1]
            if 'image_wrist' in obs:
                img = np.concatenate((img, cv2.resize(obs['image_wrist'][-1], (256, 256))), axis=1)
            img /= 255

            video.append_data(img)

        obs = copy.deepcopy(n_obs)
    if save_vid:
        video.close()

def reset_arm():
    os.system('python /home/robin/deoxys_control/deoxys/examples/reset_robot_joints.py')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, type=str, help="path to model checkpoint")
    parser.add_argument("--save_vid", action="store_true", help="create a video of rollout")
    parser.add_argument("--n_rollouts", default=10, type=int, help="number of rollouts to perform")
    args = parser.parse_args()
    
    # text = "pick up the red ball."
    # text = "pick up the green can."
    text = "place the deodorent in the pouch"
    # text = "place the can in the can holder"

    # load policy
    model = OctoModel.load_pretrained(args.ckpt)
    dataset_statistics = model.dataset_statistics

    # model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
    # # print(model.dataset_statistics.keys())
    # dataset_statistics = model.dataset_statistics['austin_sailor_dataset_converted_externally_to_rlds']
    task = model.create_tasks(texts=[text])

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
        )
    )
    
    env = FrankaGym(
                frequency=10,
                arm_enabled=True,
                cameras={
                    'primary': RealSenseCamera(serial_number='243322074131'),
                    'wrist': RealSenseCamera(serial_number='238222077150')
                }
            )
    env = EvalWrapper(env, teleop=TeleopPolicy(teleop_config))
    env = UnnormalizeActionProprio(env, dataset_statistics, normalization_type="normal")
    env = HistoryWrapper(env, 1) #Horizon
    # env = TemporalEnsembleWrapper(env, 4)
    env = RHCWrapper(env, 4)
    for _ in range(args.n_rollouts):
        rollout_policy(policy_fn, env, args.save_vid)

    env.teleop.stop()