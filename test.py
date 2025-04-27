from droid.controllers.oculus_controller import VRPolicy
from droid.robot_env import RobotEnv
from droid.user_interface.data_collector import DataCollecter

env = RobotEnv()
controller = VRPolicy(right_controller=True)
data_collector = DataCollecter(env=env, controller=controller, save_data=False)
# user_interface = RobotGUI(robot=data_collector, right_controller=True)

data_collector.collect_trajectory(practice=True)