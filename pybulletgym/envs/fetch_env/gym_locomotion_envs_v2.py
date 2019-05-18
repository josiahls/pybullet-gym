"""
Leverages existing class structure from:
https://github.com/openai/gym/blob/master/gym/envs/robotics/robot_env.py


Noting goal initialization:
initial_qpos = {
    'robot0:slide0': 0.05,
    'robot0:slide1': 0.48,
    'robot0:slide2': 0.0,
    'object0:joint': [1.7, 1.1, 0.4, 1., 0., 0., 0.],
}

.__init__(
    self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
    gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
    obj_range=0.15, target_range=0.15, distance_threshold=0.05,
    initial_qpos=initial_qpos, reward_type=reward_type)

Commit for change

Notes, the joints are labeled as follows:
'0 r_wheel_joint'
'1 l_wheel_joint'
'2 torso_lift_joint'
'3 head_pan_joint'
'4 head_tilt_joint'
'5 head_camera_joint'
'6 head_camera_rgb_joint'
'7 head_camera_rgb_optical_joint'
'8 head_camera_depth_joint'
'9 head_camera_depth_optical_joint'
'10 shoulder_pan_joint'
'11 shoulder_lift_joint'
'12 upperarm_roll_joint'
'13 elbow_flex_joint'
'14 forearm_roll_joint'
'15 wrist_flex_joint'
'16 wrist_roll_joint'
'17 gripper_axis'
'18 r_gripper_finger_joint'
'19 l_gripper_finger_joint'
'20 bellows_joint'
'21 bellows_joint2'
'22 estop_joint'
'23 laser_joint'
'24 torso_fixed_joint'
"""
import pybullet
from abc import ABC

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from pybullet_envs.bullet.bullet_client import BulletClient
from typing import Dict

from .scene_object_bases import TargetSceneObject, ProjectileSceneObject, SlicingSceneObject
from .scene_manipulators import SceneFetch, ReachScene, SlideScene, PickAndPlaceScene, KnifeScene
from .utils import Normalizer
from .env_bases import BaseBulletEnv
from .robot_locomotors import FetchURDF


class BaseFetchEnv(BaseBulletEnv, gym.GoalEnv, ABC):
    def __init__(self, initial_qpos: dict = None, robot: FetchURDF = None, block_gripper=True, n_substeps=20,
                 gripper_extra_height=0.48,
                 target_in_the_air=True, target_offset=0.0, obj_range=0.15, target_range=0.25,
                 distance_threshold=0.08, reward_type: str = 'sparse',
                 power=0.2):
        """


        Args:
            robot:
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            power (float): The amount of base power asserted by the robot on an action
        """

        self.n_substeps = n_substeps
        self.gripper_extra_height = gripper_extra_height
        self.obj_range = obj_range
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.target_range = target_range
        self.normalize_state_space_output = True
        self.np_random = None  # type: np.random.RandomState
        self.seed()
        self._p = None  # type: BulletClient
        self.goal = None
        self.reward_type = reward_type
        self.reward_threshold = -5
        self.distance_threshold = distance_threshold
        self.stateId = -1
        self.n_trackable_objects = 5
        self._use_image_state = False
        self._im_size_fac = 10  # Factor to reduce the image by
        self.initial_qpos = initial_qpos  # type: Dict[str, None]
        self.robot = None  # If not None, will be set in the parent super call

        # If true, then only the unlocked actions will be considered this means that when a model is learnign to move,
        # the expected action space output will be only the unlocked actions, as opposed to all actions.
        # This is so that we can have a model that trains faster.
        self.action_space_only_unlocked = True

        if robot is not None:
            if self.action_space_only_unlocked:
                robot.action_space_only_unlocked = self.action_space_only_unlocked
                high = np.ones([sum(~np.array(robot.lock_joints))])
                robot.action_space = gym.spaces.Box(-high, high)
                high = np.inf * np.ones([sum(~np.array(robot.lock_joints))])
                robot.observation_space = gym.spaces.Box(-high, high)

            BaseBulletEnv.__init__(self, robot)

        self.observation_space = None  # type: spaces.Dict
        # self.reset() # here

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def goal_distance(self, goal_a: np.ndarray, goal_b: np.ndarray):
        assert goal_a.shape == goal_b.shape

        distance = goal_a - goal_b
        if type(distance) is np.array:
            return np.linalg.norm(goal_a - goal_b, axis=-1)
        else:
            return np.linalg.norm(goal_a - goal_b)

    def compute_reward(self, achieved_goal: np.ndarray, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._set_action(action)
        for _ in range(self.n_substeps):
            if not self.scene.multiplayer:
                # if multiplayer, action first applied to all robots, then global step() called, then _step()
                # for all robots with the same actions
                self.scene.global_step()
            self._step_callback()

        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        Normalizer.cache_all_min_maxes()
        self._reset_sim()
        self._env_specific_callback()
        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        return obs

    def create_single_player_scene(self, _p: BulletClient):
        return SceneFetch(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)

    def close(self):
        pass

    def _reset_sim(self):
        try:
            if self._p is not None:
                self._p.getNumBodies()
        except pybullet.error:
            self.stateId = -1
            self.robot.__init__(self.robot.power, self.robot.lock_joints)
            BaseBulletEnv.__init__(self, self.robot)

        if self.action_space_only_unlocked:
            self.robot.action_space_only_unlocked = self.action_space_only_unlocked
            high = np.ones([sum(~np.array(self.robot.lock_joints))])
            self.action_space = gym.spaces.Box(-high, high)

        if self.physicsClientId < 0:
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = BulletClient()
            # noinspection PyProtectedMember
            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        # We want to clear the dynamic objects that might have been modified / added.
        # We need to do this so that we can avoid a saved state mismatch.
        if self.scene is not None:
            self.scene._actionable_object_clear()

        # The original state will only contain objects that never change i.e.
        # Never get added or removed during the course of an episode.
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)

        self.robot.reset(self._p, scene=self.scene)

        # Before adding dynamic objects, we save the original state
        if self.stateId < 0:
            self._env_setup()
            self.stateId = self._p.saveState()

        # Load dynamic objects
        self.scene.actionable_object_reset(self._p)

        return True

    def _get_obs(self):
        """
        Returns the observation.
        """
        if not self._use_image_state:
            # Get the entity state (robot usually)
            robot_state = self.robot.calc_state()
            environment_state = self.calc_state()
            self.state = np.concatenate((robot_state, environment_state), axis=1)
        else:
            self.state = np.array(self.render(mode="rgb_array") / 255)[::self._im_size_fac, ::self._im_size_fac, :]

        self.goal = self._sampled_goal_callback()
        return {
            'observation': self.state.flatten(),
            'achieved_goal': self._achieved_goal_callback(),
            'desired_goal': self._sampled_goal_callback(),
        }

    def _achieved_goal_callback(self, goal=None) -> np.array:
        """
        Default achieved goal callback is the robot's gripper position.

        Args:
            goal:

        Returns:

        """
        return self.robot.parts['gripper_link'].get_position().flatten()

    def _sampled_goal_callback(self, goal=None) -> np.array:
        """
        Default goal callback will look for a TargetObject, and
        return the object's position as the goal.

        Args:
            goal:

        Returns:

        """
        for scene_object in self.scene.scene_objects:
            if type(scene_object) is TargetSceneObject:
                return scene_object.get_position().flatten()
        return goal.flatten()

    def _env_specific_callback(self):
        for scene_object in self.scene.scene_objects:
            if type(scene_object) is ProjectileSceneObject:
                # Get the original position of the gripper
                qpos = self.robot.parts['gripper_link'].get_position()
                obj_pos = scene_object.get_position()

                while np.linalg.norm(obj_pos[:2] - qpos[:2]) < 0.1:
                    obj_pos[:2] = qpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

                scene_object.reset_position(obj_pos)

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        if not self.scene.multiplayer:
            # if multiplayer, action first applied to all robots, then global step() called, then _step()
            # for all robots with the same actions
            # self.robot.apply_action(action)
            self.robot.apply_positions(action, 0.2)

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self):
        """
        Samples a new goal and returns it.

        Originally, the random sampling as handled by the objects, however we changed this because:
        - Object's random placements are often contingent on the other objects positions to avoid collisions
            - Which means maybe random goal position should be handled by the scene?
        - The objects random position should be contingent on the robot.
            - Which means that the random position is also tied to the robot's position...

        Therefore, since the environment has information on the robot and the objects, we do the random sampling
        here. Now how much of the randomizing code needs to be here? We can A: do intelligent sampling based on
        the constrained spaces imposed by the robot and the other objects, B: do random sampling, then re-sample
        if there are collisions.

        A: would be faster
        B: would be more generalized
        """
        for scene_object in self.scene.scene_objects:
            if type(scene_object) is TargetSceneObject:
                # Get the original position of the gripper
                qpos = self.robot.parts['gripper_link'].get_position()
                # Set the position of the target relative to the gripper
                target_pos = qpos + np.random.uniform(-self.target_range, self.target_range, size=3)

                target_pos += self.target_offset

                # TODO change self.gripper_extra_height to general extra height
                # needs the motion planner for gripper extra height addition to be added.
                target_pos[2] = self.gripper_extra_height
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    target_pos[2] += self.np_random.uniform(0, 0.45)

                return scene_object.reset_position(target_pos)

    def _env_setup(self, initial_qpos=None):
        """
        Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.


        """
        if initial_qpos is None:
            initial_qpos = self.initial_qpos
            for key in [_ for _ in initial_qpos if _.__contains__('robot')]:
                self.robot.jdict[key.split(':')[-1]].set_position(initial_qpos[key], 1.2)
            for key in [_ for _ in initial_qpos if _.__contains__('TargetSceneObject')]:
                for scene_object in self.scene.scene_objects:
                    if type(scene_object) is TargetSceneObject:
                        scene_object.reset_position(initial_qpos[key][:3])
                        break
            # TODO gripper_extra_height adjust the height of the gripper.
            # Probably needs to use a deterministic motion planner.

            for _ in range(100):
                self.scene.global_step()

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def calc_state(self) -> np.array:
        """

        Each environment consists of actionable objects.
        The state for each needs to be calculated.
        The environment at max can provide the information of
        X objects as defined in the __init__ method. This is to ensure
        a common state space size.

        The environment currently loads the object states from the scene, normalizes them,
        and most importantly fills in the shape of missing objects with zeros.

        Returns: A flattened numpy array of the object state features.

        """
        object_states = self.scene.calc_state()
        if self.normalize_state_space_output:
            object_states = Normalizer.normalize(object_states, f'fetch_environment_objects_{object_states.shape}')
        # Limit the number of objects to track (Do we want to keep only objects that are of interest?)
        object_states = object_states[:self.n_trackable_objects, :]
        if self.n_trackable_objects - object_states.shape[0] > 0:
            # Fill in missing objects
            empty_placeholder = np.zeros((self.n_trackable_objects - object_states.shape[0], object_states.shape[1]))
            object_states = np.concatenate((object_states, empty_placeholder), axis=0)

        return object_states.reshape(1, -1)


class FetchReach(BaseFetchEnv, ABC):
    def __init__(self):
        robot = FetchURDF()
        robot.lock_joints = [True] * robot.action_space.shape[0]
        robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
        robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
        robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
        robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
        robot.lock_joints[17] = False  # Unlock 'gripper_axis'

        initial_qpos = {
            'robot:0:shoulder_lift_joint': -0.7,
            'robot:0:upperarm_roll_joint': 0,
            'robot:0:elbow_flex_joint': -0.9,
            'robot:0:forearm_roll_joint': 0,
            'robot:0:wrist_flex_joint': -1.35,
            'robot:0:wrist_roll_joint': 0,
            'robot:0:gripper_axis': 0,
            'TargetSceneObject:0:joint': [.75, 0, 0.60, 1., 0., 0., 0.],
        }

        super().__init__(robot=robot, initial_qpos=initial_qpos)

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = ReachScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene


class FetchSlide(BaseFetchEnv, ABC):
    def __init__(self):
        robot = FetchURDF()
        robot.lock_joints = [True] * robot.action_space.shape[0]
        robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
        robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
        robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
        robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
        robot.lock_joints[17] = False  # Unlock 'gripper_axis'

        initial_qpos = {
            'robot:0:shoulder_lift_joint': -0.7,
            'robot:0:upperarm_roll_joint': 0,
            'robot:0:elbow_flex_joint': -0.9,
            'robot:0:forearm_roll_joint': 0,
            'robot:0:wrist_flex_joint': -1.35,
            'robot:0:wrist_roll_joint': 0,
            'robot:0:gripper_axis': 0,
            'TargetSceneObject:0:joint': [1.75, 0, 0.60, 1., 0., 0., 0.],
        }

        super().__init__(robot=robot, initial_qpos=initial_qpos, target_offset=[0.4, 0, 0],
                         target_in_the_air=False, reward_type='sparse')

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = SlideScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def _achieved_goal_callback(self, goal=None) -> np.array:
        """
        Default achieved goal callback is the robot's gripper position.

        Args:
            goal:

        Returns:

        """
        for scene_object in self.scene.scene_objects:
            if type(scene_object) is ProjectileSceneObject:
                return scene_object.get_position()
        return None


class FetchPickAndPlace(BaseFetchEnv, ABC):
    def __init__(self):
        robot = FetchURDF()
        robot.lock_joints = [True] * robot.action_space.shape[0]
        robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
        robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
        robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
        robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
        robot.lock_joints[17] = False  # Unlock 'gripper_axis'

        initial_qpos = {
            'robot:0:shoulder_lift_joint': -0.7,
            'robot:0:upperarm_roll_joint': 0,
            'robot:0:elbow_flex_joint': -0.9,
            'robot:0:forearm_roll_joint': 0,
            'robot:0:wrist_flex_joint': -1.35,
            'robot:0:wrist_roll_joint': 0,
            'robot:0:gripper_axis': 0,
            'TargetSceneObject:0:joint': [0.75, 0, 0.60, 1., 0., 0., 0.],
        }

        super().__init__(robot=robot, initial_qpos=initial_qpos,
                         target_in_the_air=True, reward_type='sparse')

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = PickAndPlaceScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def _achieved_goal_callback(self, goal=None) -> np.array:
        """
        Default achieved goal callback is the robot's gripper position.

        Args:
            goal:

        Returns:

        """
        for scene_object in self.scene.scene_objects:
            if type(scene_object) is ProjectileSceneObject:
                return scene_object.get_position()
        return None


class FetchPush(BaseFetchEnv, ABC):
    def __init__(self):
        robot = FetchURDF()
        robot.lock_joints = [True] * robot.action_space.shape[0]
        robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
        robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
        robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
        robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
        robot.lock_joints[17] = False  # Unlock 'gripper_axis'

        initial_qpos = {
            'robot:0:shoulder_lift_joint': -0.7,
            'robot:0:upperarm_roll_joint': 0,
            'robot:0:elbow_flex_joint': -0.9,
            'robot:0:forearm_roll_joint': 0,
            'robot:0:wrist_flex_joint': -1.35,
            'robot:0:wrist_roll_joint': 0,
            'robot:0:gripper_axis': 0,
            'TargetSceneObject:0:joint': [0.75, 0, 0.60, 1., 0., 0., 0.],
        }

        super().__init__(robot=robot, initial_qpos=initial_qpos,
                         target_in_the_air=False, reward_type='sparse')

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = PickAndPlaceScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def _achieved_goal_callback(self, goal=None) -> np.array:
        """
        Default achieved goal callback is the robot's gripper position.

        Args:
            goal:

        Returns:

        """
        for scene_object in self.scene.scene_objects:
            if type(scene_object) is ProjectileSceneObject:
                return scene_object.get_position()
        return None


class FetchPickKnifeAndPlace(BaseFetchEnv, ABC):
    def __init__(self):
        robot = FetchURDF()
        robot.lock_joints = [True] * robot.action_space.shape[0]
        robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
        robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
        robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
        robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
        robot.lock_joints[17] = False  # Unlock 'gripper_axis'

        initial_qpos = {
            'robot:0:shoulder_lift_joint': -0.7,
            'robot:0:upperarm_roll_joint': 0,
            'robot:0:elbow_flex_joint': -0.9,
            'robot:0:forearm_roll_joint': 0,
            'robot:0:wrist_flex_joint': -1.35,
            'robot:0:wrist_roll_joint': 0,
            'robot:0:gripper_axis': 0,
            'TargetSceneObject:0:joint': [0.75, 0, 0.60, 1., 0., 0., 0.],
        }

        super().__init__(robot=robot, initial_qpos=initial_qpos,
                         target_in_the_air=False, reward_type='sparse')

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = KnifeScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def _achieved_goal_callback(self, goal=None) -> np.array:
        """
        Default achieved goal callback is the robot's gripper position.

        Args:
            goal:

        Returns:

        """
        for scene_object in self.scene.scene_objects:
            if type(scene_object) is SlicingSceneObject:
                # noinspection PyUnresolvedReferences
                return scene_object.slice_parts[0].get_position()
        return None


""" Testing Environments """


class FetchMountainCar(BaseFetchEnv, ABC):
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.position = None
        self.action_space = self.env.action_space

        super().__init__(target_in_the_air=False, reward_type='sparse')

    def reset(self):
        obs = self.env.reset()
        obs = {
            'observation': np.array(obs).reshape(1, -1),
            'achieved_goal': np.array(obs[0]),
            'desired_goal': np.array(self.env.env.goal_position),
        }
        return obs

    def render(self, mode='human'):
        if mode == 'human':
            return self.env.render(mode)
        else:
            return np.array([])

    def step(self, action):
        obs, reward, done, info = self.env.step(np.argmax(action))
        obs = {
            'observation': np.array(obs).reshape(1, -1),
            'achieved_goal': np.array(obs[0]),
            'desired_goal': np.array(self.env.env.goal_position),
        }
        info['is_success'] = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        return obs, reward, done, info
