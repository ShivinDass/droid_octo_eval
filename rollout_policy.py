from droid.controllers.oculus_controller import VRPolicy
from droid.robot_env import RobotEnv
from droid.misc.time import time_ms
import time
import numpy as np

def collect_trajectory(
        env,
        controller=None,
        policy=None,
        horizon=None,
        wait_for_controller=False,
        randomize_reset=False,
        reset_robot=True,
    ):
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

        print(obs)

        if policy is None:
            action, controller_action_info = controller.forward(obs, include_info=True)
        else:
            action = policy.forward(obs)
            controller_action_info = {}

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
            return controller_info


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

    env = RobotEnv()
    controller = VRPolicy(right_controller=True)

    collect_trajectory(
        env,
        controller=controller,
        policy=None,
        wait_for_controller=True,
        randomize_reset=False,
        reset_robot=True,
    )