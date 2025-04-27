from droid.controllers.oculus_controller import VRPolicy
from droid.robot_env import RobotEnv
from droid.misc.time import time_ms
import time
import numpy as np
from functools import partial

import jax
from octo.model.octo_model import OctoModel
import cv2

def write_video(video_frames, filename, fps=10):
    '''
    video_frames: list of frames (T, H, W, C)
    '''

    import imageio
    for i in range(len(video_frames)):
        video_frames[i] = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
    imageio.mimwrite(filename, video_frames, fps=fps)

def print_nested_keys(obs, level=0):
    for k in obs:
        if isinstance(obs[k], dict):
            print("\t"*level, k)
            print_nested_keys(obs[k], level=level+1)
        else:
            print("\t"*level, k, np.array(obs[k]).shape)

class PolicyWrapper:

    def __init__(self, policy, metadata):
        self.normalization_type = "normal"
        self.metadata = metadata

        self.policy = policy
        self.action_buffer = []

    def forward(self, obs):
        if len(self.action_buffer) == 0:
            self.processed_obs = self.process_obs(obs)
            for k in self.processed_obs.keys():
                print(k, self.processed_obs[k].shape, type(self.processed_obs[k]))

            actions = np.asarray(self.policy(self.processed_obs))
            self.action_buffer = actions.tolist()

        action = self.action_buffer.pop(0)
        action = self.process_action(action)
        return action
    
    def process_obs(self, obs):
        # Normalize Proprioception
        new_obs = {}
        # proprio = np.concatenate(obs["robot_state"]["cartesian_position"], obs["robot_state"]["gripper_"])
        # new_obs["proprio"] = self.normalize(proprio, self.metadata["proprio"])
        new_obs["image_primary"] = cv2.cvtColor(obs["image"]["36088355_left"][..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)
        new_obs["image_wrist"] = cv2.cvtColor(obs["image"]["18659563_left"][..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)
        new_obs["pad_mask"] = np.ones(1)
        return new_obs
    
    def process_action(self, action):
        # Normalize Action
        action = self.unnormalize(action, self.metadata["action"])
        action[-1] = 1-action[-1]
        action = np.clip(action, -1, 1)
        return action
    
    def unnormalize(self, data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        if self.normalization_type == "normal":
            return np.where(
                mask,
                (data * metadata["std"]) + metadata["mean"],
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
        else:
            raise ValueError(
                f"Unknown action/proprio normalization type: {self.normalization_type}"
            )

def collect_trajectory(
        env,
        controller=None,
        policy=None,
        horizon=None,
        wait_for_controller=False,
        randomize_reset=False,
        reset_robot=True,
        record_traj=False,
    ):

    recording1 = []
    recording2 = []
    # Check Parameters #
    assert (controller is not None) or (policy is not None)
    assert (controller is not None) or (horizon is not None)
    if wait_for_controller:
        assert controller is not None

    # Reset States #
    if controller is not None:
        controller.reset_state()
    env.camera_reader.set_trajectory_mode()

    # Prepare For Trajectory #
    num_steps = 0
    if reset_robot:
        env.reset(randomize=randomize_reset)

    # Begin! #
    while True:
        # Collect Miscellaneous Info #
        controller_info = {} if (controller is None) else controller.get_info()
        skip_action = wait_for_controller and (not controller_info["movement_enabled"])
        control_timestamps = {"step_start": time_ms()}

        # Get Observation #
        obs = env.get_observation()
        obs["controller_info"] = controller_info
        obs["timestamp"]["skip_action"] = skip_action
            

        action, controller_action_info = controller.forward(obs, include_info=True)
        
        if (policy is not None) and (not controller.get_info()["movement_enabled"]):
            # action = policy.forward(obs)
            action = np.zeros_like(action)
            action[0] = 0.01
            print("final action", action)
            recording1.append(policy.processed_obs["image_primary"][..., :3])
            recording2.append(policy.processed_obs["image_wrist"][..., :3])
            controller_action_info = {}
            controller.reset_state()

        # Regularize Control Frequency #
        comp_time = time_ms() - control_timestamps["step_start"]
        sleep_left = (1 / env.control_hz) - (comp_time / 1000)
        if sleep_left > 0:
            time.sleep(sleep_left)

        # Step Environment #
        control_timestamps["control_start"] = time_ms()
        if skip_action:
            action_info = env.create_action_dict(np.zeros_like(action))
        else:
            action_info = env.step(action)

        action_info.update(controller_action_info)

        # Save Data #
        control_timestamps["step_end"] = time_ms()
        obs["timestamp"]["control"] = control_timestamps
    
        # Check Termination #
        num_steps += 1
        if horizon is not None:
            end_traj = horizon == num_steps
        else:
            end_traj = controller_info["success"] or controller_info["failure"]

        # Close Files And Return #
        if end_traj:
            if record_traj:
                write_video(np.array(recording1).astype(np.uint8), 'test1.mp4', fps=10)
                write_video(np.array(recording2).astype(np.uint8), 'test2.mp4', fps=10)
            return controller_info


@jax.jit
def sample_actions(
    pretrained_model: OctoModel,
    observations,
    tasks,
    rng
):
    # add batch and horizon dim to observations
    observations = jax.tree_map(lambda x: x[None, None], observations)
    observations['pad_mask'] = observations['pad_mask'][0]
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

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, type=str, help="path to model checkpoint")
    parser.add_argument("--save_vid", action="store_true", help="create a video of rollout")
    parser.add_argument("--n_rollouts", default=10, type=int, help="number of rollouts to perform")
    args = parser.parse_args()

    text = "open the drawer"

    model = OctoModel.load_pretrained(args.ckpt)
    dataset_statistics = model.dataset_statistics

    task = model.create_tasks(texts=[text])
    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
            tasks=task,
        )
    )
    policy = PolicyWrapper(policy_fn, metadata=dataset_statistics)

    env = RobotEnv(action_space='cartesian_velocity', camera_kwargs=dict(
        hand_camera=dict(image=True, concatenate_images=False, resolution=(128, 128), resize_func="cv2"),
        varied_camera=dict(image=True, concatenate_images=False, resolution=(256, 256), resize_func="cv2"),
    ))
    controller = VRPolicy(right_controller=True)

    collect_trajectory(
        env,
        controller=controller,
        policy=policy,
        wait_for_controller=True,
        randomize_reset=False,
        reset_robot=True,
        record_traj=args.save_vid,
    )