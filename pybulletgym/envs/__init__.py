from gym.envs.registration import register

# roboschool envs
register(
	id='InvertedPendulumPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_pendulum_envs:InvertedPendulumBulletEnv',
	max_episode_steps=1000,
	reward_threshold=950.0,
	)

register(
	id='InvertedDoublePendulumPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_pendulum_envs:InvertedDoublePendulumBulletEnv',
	max_episode_steps=1000,
	reward_threshold=9100.0,
	)

register(
	id='InvertedPendulumSwingupPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_pendulum_envs:InvertedPendulumSwingupBulletEnv',
	max_episode_steps=1000,
	reward_threshold=800.0,
	)

register(
	id='ReacherPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_manipulator_envs:ReacherBulletEnv',
	max_episode_steps=150,
	reward_threshold=18.0,
	)

register(
	id='PusherPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_manipulator_envs:PusherBulletEnv',
	max_episode_steps=150,
	reward_threshold=18.0,
)

register(
	id='ThrowerPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_manipulator_envs:ThrowerBulletEnv',
	max_episode_steps=100,
	reward_threshold=18.0,
)

register(
	id='StrikerPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_manipulator_envs:StrikerBulletEnv',
	max_episode_steps=100,
	reward_threshold=18.0,
)

register(
	id='Walker2DPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_locomotion_envs:Walker2DBulletEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
	)
register(
	id='HalfCheetahPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_locomotion_envs:HalfCheetahBulletEnv',
	max_episode_steps=1000,
	reward_threshold=3000.0
	)

register(
	id='AntPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_locomotion_envs:AntBulletEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

register(
	id='HopperPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_locomotion_envs:HopperBulletEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

register(
	id='HumanoidPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_locomotion_envs:HumanoidBulletEnv',
	max_episode_steps=1000
	)

register(
	id='HumanoidFlagrunPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_locomotion_envs:HumanoidFlagrunBulletEnv',
	max_episode_steps=1000,
	reward_threshold=2000.0
	)

register(
	id='HumanoidFlagrunHarderPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_locomotion_envs:HumanoidFlagrunHarderBulletEnv',
	max_episode_steps=1000
	)

register(
	id='AtlasPyBulletEnv-v0',
	entry_point='pybulletgym.envs.roboschool.gym_locomotion_envs:AtlasBulletEnv',
	max_episode_steps=1000
	)

# mujoco envs
register(
	id='InvertedPendulumMuJoCoEnv-v0',
	entry_point='pybulletgym.envs.mujoco.gym_pendulum_envs:InvertedPendulumMuJoCoEnv',
	max_episode_steps=1000,
	reward_threshold=950.0,
)

register(
	id='InvertedDoublePendulumMuJoCoEnv-v0',
	entry_point='pybulletgym.envs.mujoco.gym_pendulum_envs:InvertedDoublePendulumMuJoCoEnv',
	max_episode_steps=1000,
	reward_threshold=9100.0,
)

register(
	id='Walker2DMuJoCoEnv-v0',
	entry_point='pybulletgym.envs.mujoco.gym_locomotion_envs:Walker2DMuJoCoEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
)
register(
	id='HalfCheetahMuJoCoEnv-v0',
	entry_point='pybulletgym.envs.mujoco.gym_locomotion_envs:HalfCheetahMuJoCoEnv',
	max_episode_steps=1000,
	reward_threshold=3000.0
)

register(
	id='AntMuJoCoEnv-v0',
	entry_point='pybulletgym.envs.mujoco.gym_locomotion_envs:AntMuJoCoEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
)

register(
	id='HopperMuJoCoEnv-v0',
	entry_point='pybulletgym.envs.mujoco.gym_locomotion_envs:HopperMuJoCoEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
)

register(
	id='HumanoidMuJoCoEnv-v0',
	entry_point='pybulletgym.envs.mujoco.gym_locomotion_envs:HumanoidMuJoCoEnv',
	max_episode_steps=1000
)

""" FETCH ENVS """
register(
	id='FetchCutBlockEnv-v1',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs:FetchCutBlockEnv',
	max_episode_steps=100000
)

# register(
# 	id='FetchReach-v0',
# 	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs:FetchReach',
# 	max_episode_steps=100000
# )

register(
	id='FetchPush-v0',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs:FetchPush',
	max_episode_steps=100000
)

register(
	id='FetchSlide-v0',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs:FetchSlide',
	max_episode_steps=100000
)

register(
	id='FetchPickAndPlace-v0',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs:FetchPickAndPlace',
	max_episode_steps=100000
)

register(
	id='FetchCutBlockNoKnifeTouchRewardEnv-v1',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs:FetchCutBlockNoKnifeTouchRewardEnv',
	max_episode_steps=100000
)

register(
	id='FetchCutBlockRandomEnv-v1',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs:FetchCutBlockEnvRandom',
	max_episode_steps=100000
)


register(
	id='FetchLiftArmHighEnv-v0',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs:FetchLiftArmHighEnv',
	max_episode_steps=100000
)

register(
	id='FetchSanityTestCartPoleEnv-v0',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs:FetchSanityTestCartPoleEnv',
	max_episode_steps=100000
)


register(
	id='FetchSanityTestMountainCar-v0',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs:FetchSanityTestMountainCar',
	max_episode_steps=100000
)

register(
	id='FetchLiftArmLowEnv-v0',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs:FetchLiftArmLowEnv',
	max_episode_steps=100000
)

register(
	id='FetchInternalKeepStillTrainEnv-v0',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs:FetchInternalKeepStillTrainEnv',
	max_episode_steps=100000
)

""" Fetch Env v2 """
register(
	id='FetchReach-v2',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs_v2:FetchReach',
	max_episode_steps=50,
)

register(
	id='FetchSlide-v2',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs_v2:FetchSlide',
	max_episode_steps=50,
)

register(
	id='FetchPickAndPlace-v2',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs_v2:FetchPickAndPlace',
	max_episode_steps=50,
)

register(
	id='FetchPickKnifeAndPlace-v2',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs_v2:FetchPickKnifeAndPlace',
	max_episode_steps=50,
)

register(
	id='FetchMountainCar-v2',
	entry_point='pybulletgym.envs.fetch_env.gym_locomotion_envs_v2:FetchMountainCar',
	max_episode_steps=50,
)


def get_list():
	envs = ['- ' + spec.id for spec in gym.pgym.envs.registry.all() if spec.id.find('Bullet') >= 0 or spec.id.find('MuJoCo') >= 0]
	return envs
