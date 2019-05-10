import pybullet
import random
from abc import ABC

import numpy as np
from pybullet_envs.bullet import bullet_client
from pybullet_envs.bullet.bullet_client import BulletClient

from .env_bases import BaseBulletEnv
from .robot_locomotors import FetchURDF
from .scene_manipulators import PickKnifeAndCutTestScene, PickAndMoveScene, KnifeCutScene, SceneFetch, ReachScene, \
    PickAndPlaceScene, SlideScene
from .scene_object_bases import Features, ProjectileSceneObject
import os
"""
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


class BaseFetchEnv(BaseBulletEnv, ABC):
    def __init__(self, is_sanity_test=False, distance_threshold = 0, reward_type='default', power=0.2):
        """
        BaseFetchEnv is concerned with the following:
        - Fetch robot complexity
        - Morphological object change
        - Interactive object behavior

        The point of this base class is due to the massive number of envs that
        will be branching from using a fetch. We need to stream-line the
        environment creation and testing. Most of all, reduce threat of
        bugs crashing an env training 50% of the way through.
        """
        self.power = power
        if not is_sanity_test:
            self.robot = FetchURDF(self.power)
            BaseBulletEnv.__init__(self, self.robot)

        self.joints_at_limit_cost = -1.
        self.scene = None
        self.potential = 0
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.rewards = []

        self.joints_are_locked = False

        # Handles augmenting the rewards if the robot is failing / succeeding too much
        self._total_rewards = [0.0]
        self._accumulated_total_rewards = [0.0]
        self._should_throw_bone = False
        self._neg_bone = 1.0
        self._pos_bone = 1.0
        self._bone_max = 5.0
        self._pos_bone_min = 0.1
        self._bone_gamma = 0.001
        self._percent_failure_allowable = 0.9
        self._percent_success_allowable = 0.2
        self._percent_margin = 0.1
        self._do_reward_balancing = False

        self.stateId = -1
        self.frame = 0
        self.done = 0
        self.reward = 0.0
        self.elapsed_time = 0
        self.elapsed_time_cost = 0.01
        self.max_step_length = 30
        self.min_step_length = 20
        self.max_state_space_object_size = 3
        self.use_image_as_state = False
        self.state = None
        self._cam_yaw = 90
        self._p = None

        self.use_normalization = True
        self.state_min_maxes = {}  # type: dict

    def reset(self):
        """
        More Fetch specific reset functionality.

        One of the biggest differences is removing adding a scene to the robot.
        The goal here is to reduce the number of instances of a scene object to 1.

        :return:
        """
        print(f'Setting Environment: Doing Reward Aug? {self._do_reward_balancing} Doing joint locking? '
              f'{self.joints_are_locked}')
        for key in self.state_min_maxes:
            # Save the normalization fields:
            np.save(key, self.state_min_maxes[key])

        if self.physicsClientId < 0:
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self._p = bullet_client.BulletClient()

            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        if self.scene is None:
            self.scene = self.create_single_player_scene(self._p)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        # We want to clear the dynamic objects that might have been modified / added.
        # We need to do this so that we can avoid a saved state mismatch.
        if self.scene is not None:
            self.scene._dynamic_object_clear()

        # The original state will only contain objects that never change i.e.
        # Never get added or removed during the course of an episode.
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)

        self.frame = 0
        self.done = 0
        self.reward = 0.0
        self.elapsed_time = 0
        self._accumulated_total_rewards.append(np.average(self._total_rewards))
        self._total_rewards = [0.0]
        s = self.robot.reset(self._p, scene=self.scene)
        self.robot.robot_specific_reset(self._p)
        self.camera._p = self._p
        self.potential = self.robot.calc_potential(scene=self.scene)

        # Before adding dynamic objects, we save the original state
        if self.stateId < 0:
            self.stateId = self._p.saveState()

        # Load dynamic objects
        self.scene.dynamic_object_reset(self._p)

        # Update the robot init position if available
        # while not self.init_robot_pose():
        #     pass

        return self.get_full_state()

    def create_single_player_scene(self, _p: BulletClient):
        return SceneFetch(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)

    def camera_adjust(self):
        x, y, z = self.robot.body_xyz
        self.camera_x = 0.98 * self.camera_x + (1 - 0.98) * x
        self.camera.move_and_look_at(self.camera_x, y - 2.0, 1.4, x, y, 1.0)

    def get_full_state(self, use_image=False) -> np.array:
        """
        Returns a singular state space 1D representation of the environment's space.

        The base env has a fixed limit of number of objects to track, and so if the number
        of objects < max_state_space_object_size then the remaining space is filled with zeros

        If image is None (default) then the environment will use it's default state space representation.
        This representation will be a combination of joint, object, and robot features.

        Args:
            image: If not None, will be used for the state space.

        Returns:

        """
        self.use_image_as_state = use_image
        if not self.use_image_as_state:
            # Get the state of the robot
            state = self.robot.calc_state().reshape((1, -1))
            object_states = self.scene.calc_state()

            if self.use_normalization:
                state = self._normalize_states(state, 'robot_state_min_max.npy')
                object_states = self._normalize_states(state, 'object_states_min_max.npy')

            for i, object_state in enumerate(object_states):
                if i < self.max_state_space_object_size:
                    state = np.hstack((state, np.array(object_state).reshape((1, -1))))

            for i in range(self.max_state_space_object_size - len(object_states)):
                state = np.hstack((state, np.zeros((1, len(Features())))))

            self.state = state
        else:
            self.state = np.array(self.render(mode="rgb_array") / 255)[::10, ::10, :]

        return self.state

    def _normalize_states(self, state, filename='robot_state_min_max.npy'):
        if not os.path.exists(os.path.join('.', 'normalization_weights')):
            os.mkdir(os.path.join('.', 'normalization_weights'))

        filename = os.path.join('.', 'normalization_weights', filename)
        # Check if there exist cached min max information for normalization
        if filename not in self.state_min_maxes and os.path.exists(filename):
            self.state_min_maxes[filename] = np.load(filename)

        # Init the fields if needed
        if filename not in self.state_min_maxes:
            self.state_min_maxes[filename] = np.zeros(np.array(state).shape)
            self.state_min_maxes[filename] = np.vstack((self.state_min_maxes[filename], np.min(state, axis=0)))
        else:
            # If it is not none, look at the robot state, the current min max
            min_max_slice = np.vstack((state, self.state_min_maxes[filename]))
            self.state_min_maxes[filename][0] = np.max(min_max_slice, axis=0)
            self.state_min_maxes[filename][1] = np.min(min_max_slice, axis=0)

        # Normalize the states
        state = (state - self.state_min_maxes[filename][1]) / \
                    (self.state_min_maxes[filename][0] - self.state_min_maxes[filename][1] + 0.001)
        return state

    def step(self, a):
        """ UPDATE ACTIONS """
        if not self.scene.multiplayer:
            # if multiplayer, action first applied to all robots, then global step() called, then _step()
            # for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        """ CALCULATE STATES """
        # also calculates self.joints_at_limit
        self.state = self.get_full_state(self.use_image_as_state)

        """ CALCULATE REWARDS (internal to the robot)"""
        # For no, the robot will always be alive
        # Otherwise, if the robot is upright, reward it
        # alive = float(self.robot.alive_bonus(state[0][0] + self.robot.initial_z, self.robot.body_rpy))
        alive = float(self.robot.alive_bonus(self.robot.initial_z, self.robot.body_rpy))

        # Punish higher amounts of time
        self.elapsed_time += 1
        time_cost = self.elapsed_time_cost if self.elapsed_time > self.min_step_length else 0

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        joints_at_speed_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_speed_limit)
        debugmode = 0
        if debugmode:
            print("alive=")
            print(alive)
            # print("progress")
            # print(progress)
            # print("electricity_cost")
            # print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)

        # Get the env custom reward
        custom_reward = self.get_custom_reward()

        self.rewards = [
            alive,
            -1 * self.elapsed_time * time_cost,
            joints_at_limit_cost,
            joints_at_speed_limit_cost,
            custom_reward,
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))

        # If the robot has died, and the elasped
        done = alive < 0 and self.elapsed_time > self.min_step_length
        if not np.isfinite(self.state).all():
            print("~INF~", self.state)
            done = True
        if done:
            print(f'Done because: state[0] is {self.state[0][0]} and the initial z is: {self.robot.initial_z} and the rpy '
                  f'rpy is {self.robot.body_rpy} rxy is {self.robot.body_xyz}')
        if not done and self.elapsed_time > self.max_step_length:
            done = True

        if alive < 0:
            # If the robot is no longer alive due to falling over, then ensure that we return the worst possible reward
            self.rewards = [min(self._accumulated_total_rewards)]

        self.HUD(self.state, a, done)
        self._total_rewards.append(sum(self.rewards))

        return self.state, sum(self.rewards), bool(done), {}

    def get_entropy_state(self):
        return self.robot.gripper_link.get_position()

    def get_custom_reward(self, achieved_goal=None, goal=None):
        return 0

    def init_robot_pose(self) -> bool:
        return True

    def execute_joint_lock(self, should_lock=None):
        """
        Each environment has the option to lock actions. This provides an easy way to
        lock and unlock those actions.

        :return:
        """
        pass

    def _goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def get_achieved_goal(self):
        pass


class FetchReach(BaseFetchEnv, ABC):
    def __init__(self, is_sanity_test=False, distance_threshold = 0, reward_type='default', power=0.2):
        super().__init__(is_sanity_test, distance_threshold, reward_type, power)

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = ReachScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def get_custom_reward(self, achieved_goal=None, goal=None):
        if achieved_goal is None:
            achieved_goal = self.get_achieved_goal()

        if goal is None:
            goal = self.scene.get_goal()[0]
        
        assert achieved_goal is not None, 'The achieved_goal is None. Needs to be an np.array'
        assert goal is not None, 'The goal is None. Needs to be an np.array'

        d = self._goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def execute_joint_lock(self, should_lock=None):
        if should_lock is not None and type(should_lock) is bool:
            self.joints_are_locked = should_lock
        if self.joints_are_locked:
            self.robot.lock_joints = [True] * self.action_space.shape[0]
            self.robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
            self.robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
            # self.robot.lock_joints[12] = False  # Unlock 'upperarm_roll_joint'
            self.robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
            # self.robot.lock_joints[14] = False  # Unlock 'forearm_roll_joint'
            self.robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
            # self.robot.lock_joints[16] = False  # Unlock 'wrist_roll_joint'
            self.robot.lock_joints[17] = False  # Unlock 'gripper_axis'
            self.robot.lock_joints[18] = False  # Unlock 'r_gripper_finger_joint'
            self.robot.lock_joints[19] = False  # Unlock 'l_gripper_finger_joint'
        else:
            self.robot.lock_joints = [False] * self.action_space.shape[0]

        self.joints_are_locked = not self.joints_are_locked

    def get_achieved_goal(self):
        return np.average((self.robot.l_gripper_finger_link.get_position(),
                           self.robot.r_gripper_finger_link.get_position()), axis=0)


class FetchPickAndPlace(BaseFetchEnv, ABC):
    def __init__(self, is_sanity_test=False, distance_threshold=0, reward_type='default', power=0.2):
        super().__init__(is_sanity_test, distance_threshold, reward_type, power)

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = PickAndPlaceScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def get_custom_reward(self, achieved_goal=None, goal=None):
        if achieved_goal is None:
            achieved_goal = self.get_achieved_goal()

        if goal is None:
            goal = self.scene.get_goal()[0]

        assert achieved_goal is not None, 'The achieved_goal is None. Needs to be an np.array'
        assert goal is not None, 'The goal is None. Needs to be an np.array'

        d = self._goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def get_achieved_goal(self):
        return np.average(tuple([scene_object.get_position() for scene_object in self.scene.scene_objects
                                 if type(scene_object) is ProjectileSceneObject]), axis=0)

    def execute_joint_lock(self, should_lock=None):
        if should_lock is not None and type(should_lock) is bool:
            self.joints_are_locked = should_lock
        if self.joints_are_locked:
            self.robot.lock_joints = [True] * self.action_space.shape[0]
            self.robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
            self.robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
            # self.robot.lock_joints[12] = False  # Unlock 'upperarm_roll_joint'
            self.robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
            # self.robot.lock_joints[14] = False  # Unlock 'forearm_roll_joint'
            self.robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
            # self.robot.lock_joints[16] = False  # Unlock 'wrist_roll_joint'
            self.robot.lock_joints[17] = False  # Unlock 'gripper_axis'
            self.robot.lock_joints[18] = False  # Unlock 'r_gripper_finger_joint'
            self.robot.lock_joints[19] = False  # Unlock 'l_gripper_finger_joint'
        else:
            self.robot.lock_joints = [False] * self.action_space.shape[0]

        self.joints_are_locked = not self.joints_are_locked


class FetchPush(BaseFetchEnv, ABC):
    def __init__(self, is_sanity_test=False, distance_threshold=0, reward_type='default', power=0.2):
        super().__init__(is_sanity_test, distance_threshold, reward_type, power)

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = PickAndPlaceScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def get_custom_reward(self, achieved_goal=None, goal=None):
        if achieved_goal is None:
            achieved_goal = self.get_achieved_goal()

        if goal is None:
            goal = self.scene.get_goal()[0]

        assert achieved_goal is not None, 'The achieved_goal is None. Needs to be an np.array'
        assert goal is not None, 'The goal is None. Needs to be an np.array'

        d = self._goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def get_achieved_goal(self):
        return np.average(tuple([scene_object.get_position() for scene_object in self.scene.scene_objects
                                 if type(scene_object) is ProjectileSceneObject]), axis=0)

    def execute_joint_lock(self, should_lock=None):
        if should_lock is not None and type(should_lock) is bool:
            self.joints_are_locked = should_lock
        if self.joints_are_locked:
            self.robot.lock_joints = [True] * self.action_space.shape[0]
            self.robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
            self.robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
            # self.robot.lock_joints[12] = False  # Unlock 'upperarm_roll_joint'
            self.robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
            # self.robot.lock_joints[14] = False  # Unlock 'forearm_roll_joint'
            self.robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
            # self.robot.lock_joints[16] = False  # Unlock 'wrist_roll_joint'
            self.robot.lock_joints[17] = False  # Unlock 'gripper_axis'
            # self.robot.lock_joints[18] = False  # Unlock 'r_gripper_finger_joint'
            # self.robot.lock_joints[19] = False  # Unlock 'l_gripper_finger_joint'
        else:
            self.robot.lock_joints = [False] * self.action_space.shape[0]

        self.joints_are_locked = not self.joints_are_locked


class FetchSlide(BaseFetchEnv, ABC):
    def __init__(self, is_sanity_test=False, distance_threshold=0, reward_type='default', power=0.2):
        super().__init__(is_sanity_test, distance_threshold, reward_type, power)

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = SlideScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def get_custom_reward(self, achieved_goal=None, goal=None):
        if achieved_goal is None:
            achieved_goal = self.get_achieved_goal()

        if goal is None:
            goal = self.scene.get_goal()[0]

        assert achieved_goal is not None, 'The achieved_goal is None. Needs to be an np.array'
        assert goal is not None, 'The goal is None. Needs to be an np.array'

        d = self._goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def get_achieved_goal(self):
        return np.average(tuple([scene_object.get_position() for scene_object in self.scene.scene_objects
                                 if type(scene_object) is ProjectileSceneObject]), axis=0)

    def execute_joint_lock(self, should_lock=None):
        if should_lock is not None and type(should_lock) is bool:
            self.joints_are_locked = should_lock
        if self.joints_are_locked:
            self.robot.lock_joints = [True] * self.action_space.shape[0]
            self.robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
            self.robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
            # self.robot.lock_joints[12] = False  # Unlock 'upperarm_roll_joint'
            self.robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
            # self.robot.lock_joints[14] = False  # Unlock 'forearm_roll_joint'
            self.robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
            # self.robot.lock_joints[16] = False  # Unlock 'wrist_roll_joint'
            self.robot.lock_joints[17] = False  # Unlock 'gripper_axis'
            # self.robot.lock_joints[18] = False  # Unlock 'r_gripper_finger_joint'
            # self.robot.lock_joints[19] = False  # Unlock 'l_gripper_finger_joint'
        else:
            self.robot.lock_joints = [False] * self.action_space.shape[0]

        self.joints_are_locked = not self.joints_are_locked


class FetchSanityTestCartPoleEnv(BaseFetchEnv, ABC):

    def __init__(self, is_sanity_test=False, distance_threshold = 0, reward_type='default', power=0.2):
        super().__init__(is_sanity_test, distance_threshold, reward_type, power)
        """
        Wraps the BaseFetchEnv but for the cart pole env.

        Will be important for knowing that a certain model converges.
        Needs to subclass most of the parent method calls since the robot will not exist
        """
        import gym
        self.internal_env = gym.make('CartPole-v0')
        self.action_space = np.zeros((self.internal_env.action_space.n, 1))
        self.state = None

    def create_single_player_scene(self, _p: BulletClient):
        return None

    def reset(self):
        self.frame = 0
        self.done = 0
        self.reward = 0.0
        self.elapsed_time = 0
        self._accumulated_total_rewards.append(np.average(self._total_rewards))
        self._total_rewards = [0.0]
        self.state = self.internal_env.reset()
        self.state = self.get_full_state(self.use_image_as_state)

        print(f'Setting Environment: Doing Reward Aug? {self._do_reward_balancing} Doing joint locking? '
              f'{self.joints_are_locked}')
        for key in self.state_min_maxes:
            # Save the normalization fields:
            np.save(key, self.state_min_maxes[key])

        return self.state

    def get_entropy_state(self):

        if self.use_image_as_state:
            state = [np.array(self.state).flatten().transpose(0)]
        else:
            state = np.reshape(self.state, [1, self.internal_env.observation_space.shape[0]])
        return state[0]

    def camera_adjust(self):
        pass

    def render(self, mode, **kwargs):
        return self.internal_env.render(mode)

    def get_full_state(self, use_image=False) -> np.array:

        if not use_image:
            state = np.reshape(self.state, [1, self.internal_env.observation_space.shape[0]])
        else:
            state = super(FetchSanityTestCartPoleEnv, self).get_full_state(use_image)

        if self.use_normalization:
            state = self._normalize_states(state, 'cart_state_min_max.npy')

        print(f'Current State {np.array([np.array([state[0][3]])])}')
        # return np.array(state)
        return state

    def step(self, a):
        """ UPDATE ACTIONS """
        self.state, reward, done, info = self.internal_env.step(np.argmax(a))
        state = self.get_full_state(self.use_image_as_state)
        return state, reward, done, info


class FetchSanityTestMountainCar(BaseFetchEnv, ABC):

    def __init__(self, is_sanity_test=False, distance_threshold = 0, reward_type='default', power=0.2):
        super().__init__(is_sanity_test, distance_threshold, reward_type, power)
        """
        Wraps the BaseFetchEnv but for the cart pole env.

        Will be important for knowing that a certain model converges.
        Needs to subclass most of the parent method calls since the robot will not exist
        """
        import gym
        self.internal_env = gym.make('MountainCar-v0')
        self.action_space = np.zeros((self.internal_env.action_space.n, 1))
        self.state = None

    def create_single_player_scene(self, _p: BulletClient):
        return None

    def reset(self):
        self.frame = 0
        self.done = 0
        self.reward = 0.0
        self.elapsed_time = 0
        self._accumulated_total_rewards.append(np.average(self._total_rewards))
        self._total_rewards = [0.0]
        self.state = self.internal_env.reset()
        self.state = np.reshape(self.state, [1, self.internal_env.observation_space.shape[0]])

        print(f'Setting Environment: Doing Reward Aug? {self._do_reward_balancing} Doing joint locking? '
              f'{self.joints_are_locked}')
        for key in self.state_min_maxes:
            # Save the normalization fields:
            np.save(key, self.state_min_maxes[key])

        return self.state

    def get_entropy_state(self):

        if self.use_image_as_state:
            state = [np.array(self.state).flatten().transpose(0)]
        else:
            state = np.reshape(self.state, [1, self.internal_env.observation_space.shape[0]])
        return state[0]

    def camera_adjust(self):
        pass

    def render(self, mode, **kwargs):
        return self.internal_env.render(mode)

    def get_full_state(self, use_image=False) -> np.array:

        if not use_image:
            state = np.reshape(self.state, [1, self.internal_env.observation_space.shape[0]])
        else:
            state = super(FetchSanityTestMountainCar, self).get_full_state(use_image)

        if self.use_normalization:
            state = self._normalize_states(state, 'mountain_car_min_max.npy')

        # return np.hstack((np.zeros(state.shape), state))
        return state

    def step(self, a):
        """ UPDATE ACTIONS """
        self.state, reward, done, info = self.internal_env.step(np.argmax(a))
        state = self.get_full_state(self.use_image_as_state)
        return state, reward, done, info


class FetchMoveBlockEnv(BaseFetchEnv, ABC):

    def __init__(self, is_sanity_test=False, distance_threshold = 0, reward_type='default', power=0.2):
        super().__init__(is_sanity_test, distance_threshold, reward_type, power)
        self.init_positions = [0] * self.action_space.shape[0]
        self.init_positions[11] = -0.2
        self._index = 0

    def init_robot_pose(self):
        """ UPDATE ACTIONS """
        if not self.scene.multiplayer:
            # if multiplayer, action first applied to all robots, then global step() called, then _step()
            # for all robots with the same actions
            self.robot.apply_positions(self.init_positions, 0.9)
            self._index += 1
        self.scene.global_step()

        """ CALCULATE STATES """
        # also calculates self.joints_at_limit
        self.get_full_state()

        if self._index > 50:
            self._index = 0
            return True
        else:
            return False

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = PickAndMoveScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def execute_joint_lock(self, should_lock=None):
        if should_lock is not None and type(should_lock) is bool:
            self.joints_are_locked = should_lock

        if self.joints_are_locked:
            self.robot.lock_joints = [True] * self.action_space.shape[0]
            self.robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
            self.robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
            self.robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
            self.robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
            self.robot.lock_joints[18] = False  # Unlock 'r_gripper_finger_joint'
            self.robot.lock_joints[19] = False  # Unlock 'l_gripper_finger_joint'
        else:
            self.robot.lock_joints = [False] * self.action_space.shape[0]

        self.joints_are_locked = not self.joints_are_locked


class FetchCutBlockEnvRandom(BaseFetchEnv, ABC):
    def __init__(self, is_sanity_test=False, distance_threshold = 0, reward_type='default', power=0.2):
        super().__init__(is_sanity_test, distance_threshold, reward_type, power)
        """
        Rewards the robot to cutting a block.
        Inits the arm to be grasping the knife.
        """

        self.init_positions = [0] * self.action_space.shape[0]
        self.init_positions[11] = -0.72
        self.init_positions[12] = -0.2
        self.init_positions[13] = -0.9
        self.init_positions[15] = -1.4

        self._index = 0

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = KnifeCutScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def init_robot_pose(self):
        """ UPDATE ACTIONS """
        if not self.scene.multiplayer:
            # if multiplayer, action first applied to all robots, then global step() called, then _step()
            # for all robots with the same actions
            self.robot.apply_positions(self.init_positions, 0.9)
            self._index += 1
        self.scene.global_step()

        """ CALCULATE STATES """
        # also calculates self.joints_at_limit
        self.get_full_state()

        if self._index < 5:
            self.init_positions[13] = -0.9

        if self._index > 160:
            self.init_positions[18] = 0.015
            self.init_positions[19] = 0.015
        else:
            self.init_positions[18] = 0.05
            self.init_positions[19] = 0.05

        if self._index > 180:
            self.init_positions[13] = -0.7

        if self._index > 210:
            self._index = 0
            return True
        else:
            return False

    def get_custom_reward(self, achieved_goal=None, goal=None):
        """
        The reward for this function is increasing the number of scene objects, distance fingers to knife, touching the
        knife, moving the knife closer to the object.

        Returns:
        """
        # Needs more than 2 objects to get reward
        n_scene_objects = len(self.scene.scene_objects) - 2

        # Inverse distance to the knife
        if self.scene.scene_objects is None:
            return 0

        target_positions = [_.get_position() for _ in self.scene.scene_objects if
                            _.filename is not None and _.filename.__contains__('knife.urdf')]
        inv_distance = 2 - np.linalg.norm(np.subtract(target_positions,
                                                      [self.robot.l_gripper_finger_link.get_position(),
                                                       self.robot.r_gripper_finger_link.get_position()]))

        # Is Touching the knife
        is_touching = int(inv_distance < .1)

        inv_distance_to_cube = 0

        # Knife is closer to the object
        for scene_object in self.scene.scene_objects:
            if scene_object.filename.__contains__('cube_concave.urdf'):
                inv_distance_to_cube += 1 - np.linalg.norm(np.subtract(target_positions, scene_object.get_position()))

        return sum([n_scene_objects, inv_distance, inv_distance_to_cube, is_touching])

    def execute_joint_lock(self, should_lock=None):
        if should_lock is not None and type(should_lock) is bool:
            self.joints_are_locked = should_lock
        if self.joints_are_locked:
            self.robot.lock_joints = [True] * self.action_space.shape[0]
            self.robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
            self.robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
            self.robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
            # self.robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
            self.robot.lock_joints[18] = False  # Unlock 'r_gripper_finger_joint'
            self.robot.lock_joints[19] = False  # Unlock 'l_gripper_finger_joint'
        else:
            self.robot.lock_joints = [False] * self.action_space.shape[0]

        self.joints_are_locked = not self.joints_are_locked


class FetchCutBlockEnv(BaseFetchEnv, ABC):
    def __init__(self, is_sanity_test=False, distance_threshold = 0, reward_type='default', power=0.2):
        super().__init__(is_sanity_test, distance_threshold, reward_type, power)
        """
        Rewards the robot to cutting a block.
        Inits the arm to be grasping the knife.
        """

        self.init_positions = [0] * self.action_space.shape[0]
        self.init_positions[11] = -0.72
        self.init_positions[12] = -0.2
        self.init_positions[13] = -0.9
        self.init_positions[15] = -1.4

        self._index = 0

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = KnifeCutScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4, randomize=True)
        return self.scene

    def init_robot_pose(self):
        """ UPDATE ACTIONS """
        if not self.scene.multiplayer:
            # if multiplayer, action first applied to all robots, then global step() called, then _step()
            # for all robots with the same actions
            self.robot.apply_positions(self.init_positions, 0.9)
            self._index += 1
        self.scene.global_step()

        """ CALCULATE STATES """
        # also calculates self.joints_at_limit
        self.get_full_state()

        if self._index < 5:
            self.init_positions[13] = -0.9

        if self._index > 160:
            self.init_positions[18] = 0.015
            self.init_positions[19] = 0.015
        else:
            self.init_positions[18] = 0.05
            self.init_positions[19] = 0.05

        if self._index > 180:
            self.init_positions[13] = -0.7

        if self._index > 210:
            self._index = 0
            return True
        else:
            return False

    def get_custom_reward(self, achieved_goal=None, goal=None):
        """
        The reward for this function is increasing the number of scene objects, distance fingers to knife, touching the
        knife, moving the knife closer to the object.

        Returns:
        """
        # Needs more than 2 objects to get reward
        n_scene_objects = len(self.scene.scene_objects) - 2

        # Inverse distance to the knife
        if self.scene.scene_objects is None:
            return 0

        target_positions = [_.get_position() for _ in self.scene.scene_objects if
                            _.filename is not None and _.filename.__contains__('knife.urdf')]
        inv_distance = 2 - np.linalg.norm(np.subtract(target_positions,
                                                      [self.robot.l_gripper_finger_link.get_position(),
                                                       self.robot.r_gripper_finger_link.get_position()]))

        # Is Touching the knife
        is_touching = int(inv_distance < .1)

        inv_distance_to_cube = 0

        # Knife is closer to the object
        for scene_object in self.scene.scene_objects:
            if scene_object.filename.__contains__('cube_concave.urdf'):
                inv_distance_to_cube += 1 - np.linalg.norm(np.subtract(target_positions, scene_object.get_position()))

        return sum([n_scene_objects, inv_distance, inv_distance_to_cube, is_touching])

    def execute_joint_lock(self, should_lock=None):
        if should_lock is not None and type(should_lock) is bool:
            self.joints_are_locked = should_lock
        if self.joints_are_locked:
            self.robot.lock_joints = [True] * self.action_space.shape[0]
            self.robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
            self.robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
            self.robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
            # self.robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
            self.robot.lock_joints[18] = False  # Unlock 'r_gripper_finger_joint'
            self.robot.lock_joints[19] = False  # Unlock 'l_gripper_finger_joint'
        else:
            self.robot.lock_joints = [False] * self.action_space.shape[0]

        self.joints_are_locked = not self.joints_are_locked


class FetchCutBlockNoKnifeTouchRewardEnv(BaseFetchEnv, ABC):
    def __init__(self, is_sanity_test=False, distance_threshold = 0, reward_type='default', power=0.2):
        super().__init__(is_sanity_test, distance_threshold, reward_type, power)
        """
        Rewards the robot to cutting a block.
        Inits the arm to be grasping the knife.
        """

        self.init_positions = [0] * self.action_space.shape[0]
        self.init_positions[11] = -0.72
        self.init_positions[12] = -0.2
        self.init_positions[13] = -0.9
        self.init_positions[15] = -1.4

        self._index = 0

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = KnifeCutScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4, randomize=True)
        return self.scene

    def init_robot_pose(self):
        """ UPDATE ACTIONS """
        if not self.scene.multiplayer:
            # if multiplayer, action first applied to all robots, then global step() called, then _step()
            # for all robots with the same actions
            self.robot.apply_positions(self.init_positions, 0.9)
            self._index += 1
        self.scene.global_step()

        """ CALCULATE STATES """
        # also calculates self.joints_at_limit
        self.get_full_state()

        if self._index < 5:
            self.init_positions[13] = -0.9

        if self._index > 160:
            self.init_positions[18] = 0.015
            self.init_positions[19] = 0.015
        else:
            self.init_positions[18] = 0.05
            self.init_positions[19] = 0.05

        if self._index > 180:
            self.init_positions[13] = -0.7

        if self._index > 210:
            self._index = 0
            return True
        else:
            return False

    def get_custom_reward(self, achieved_goal=None, goal=None):
        """
        The reward for this function is increasing the number of scene objects, distance fingers to knife, touching the
        knife, moving the knife closer to the object.

        Returns:
        """
        # Needs more than 2 objects to get reward
        n_scene_objects = len(self.scene.scene_objects) - 2

        # Inverse distance to the knife
        if self.scene.scene_objects is None:
            return 0

        target_positions = [_.get_position() for _ in self.scene.scene_objects if
                            _.filename is not None and _.filename.__contains__('knife.urdf')]
        inv_distance = 2 - np.linalg.norm(np.subtract(target_positions,
                                                      [self.robot.l_gripper_finger_link.get_position(),
                                                       self.robot.r_gripper_finger_link.get_position()]))

        # Is Touching the knife
        is_touching = 0  # int(inv_distance < .1)

        inv_distance_to_cube = 0

        # Knife is closer to the object
        for scene_object in self.scene.scene_objects:
            if scene_object.filename.__contains__('cube_concave.urdf'):
                inv_distance_to_cube += 1 - np.linalg.norm(np.subtract(target_positions, scene_object.get_position()))

        return sum([n_scene_objects, inv_distance, inv_distance_to_cube, is_touching])

    def execute_joint_lock(self, should_lock=None):
        if should_lock is not None and type(should_lock) is bool:
            self.joints_are_locked = should_lock
        if self.joints_are_locked:
            self.robot.lock_joints = [True] * self.action_space.shape[0]
            self.robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
            self.robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
            self.robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
            # self.robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
            self.robot.lock_joints[18] = False  # Unlock 'r_gripper_finger_joint'
            self.robot.lock_joints[19] = False  # Unlock 'l_gripper_finger_joint'
        else:
            self.robot.lock_joints = [False] * self.action_space.shape[0]

        self.joints_are_locked = not self.joints_are_locked


class FetchLiftArmHighEnv(BaseFetchEnv, ABC):
    def __init__(self, is_sanity_test=False, distance_threshold = 0, reward_type='default', power=0.2):
        super().__init__(is_sanity_test, distance_threshold, reward_type, power)
        """
        Rewards the robot to lifting its arm high.
        """
        self.max_ceiling = 2
        self.min_floor = 0.5
        self.randomCeiling = random.uniform(self.min_floor, self.max_ceiling)

        self.init_positions = [0] * self.action_space.shape[0]
        self._index = 0

    def init_robot_pose(self):
        """ UPDATE ACTIONS """
        if not self.scene.multiplayer:
            # if multiplayer, action first applied to all robots, then global step() called, then _step()
            # for all robots with the same actions
            self.robot.apply_positions(self.init_positions, 0.9)
            self._index += 1
        self.scene.global_step()

        if self._index <= 1:
            self.init_positions[10] = random.uniform(-1.3, 1.3)  # shoulder_pan_joint
            self.init_positions[11] = random.uniform(-0.72, 0)  # shoulder_lift_joint
            self.init_positions[12] = random.uniform(-0.4, 0.4)  # upperarm_roll_joint
            self.init_positions[13] = random.uniform(0.0, 0.9)  # elbow_flex_joint

        """ CALCULATE STATES """
        # also calculates self.joints_at_limit
        self.get_full_state()

        if self._index > 180:
            self._index = 0
            return True
        else:
            return False

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = PickAndMoveScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def get_custom_reward(self, achieved_goal=None, goal=None):

        # Distance to the arm goal
        distance = np.linalg.norm(np.subtract([self.randomCeiling, self.randomCeiling],
                                              [self.robot.l_gripper_finger_link.get_position()[2],
                                               self.robot.r_gripper_finger_link.get_position()[2]]))
        # Punish or reward?
        sign = 5 if distance < (self.max_ceiling - self.min_floor) else -1

        return abs(1 - distance / self.max_ceiling) * sign

    def execute_joint_lock(self, should_lock=None):
        if should_lock is not None and type(should_lock) is bool:
            self.joints_are_locked = should_lock

        if self.joints_are_locked:
            self.robot.lock_joints = [True] * self.action_space.shape[0]
            self.robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
            self.robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
            self.robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
            self.robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
        else:
            self.robot.lock_joints = [False] * self.action_space.shape[0]

        self.joints_are_locked = not self.joints_are_locked


class FetchLiftArmLowEnv(BaseFetchEnv, ABC):
    def __init__(self, is_sanity_test=False, distance_threshold = 0, reward_type='default', power=0.2):
        super().__init__(is_sanity_test, distance_threshold, reward_type, power)
        """
        The reward functions for this env will involve lifting the arm to the ground.
        It is expected the the robot will try to move the arm around the table.
        """

        self._index = 0
        self.max_ceiling = 0.7
        self.min_floor = 0
        self.randomCeiling = random.uniform(self.min_floor, self.max_ceiling)

        self.init_positions = [0] * self.action_space.shape[0]

    def init_robot_pose(self):
        """ UPDATE ACTIONS """
        if not self.scene.multiplayer:
            # if multiplayer, action first applied to all robots, then global step() called, then _step()
            # for all robots with the same actions
            self.robot.apply_positions(self.init_positions, 0.9)
            self._index += 1
        self.scene.global_step()

        """ CALCULATE STATES """
        # also calculates self.joints_at_limit
        self.get_full_state()

        if self._index <= 1:
            self.init_positions[10] = random.uniform(-1.3, 1.3)  # shoulder_pan_joint
            self.init_positions[11] = random.uniform(-0.72, 0)  # shoulder_lift_joint
            self.init_positions[12] = random.uniform(-0.4, 0.4)  # upperarm_roll_joint
            self.init_positions[13] = random.uniform(0.0, 0.9)  # elbow_flex_joint

        if self._index > 180:
            self._index = 0
            return True
        else:
            return False

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = PickAndMoveScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def get_custom_reward(self, achieved_goal=None, goal=None):
        # Distance to the arm goal
        distance = np.linalg.norm(np.subtract([self.randomCeiling, self.randomCeiling],
                                              [self.robot.l_gripper_finger_link.get_position()[2],
                                               self.robot.r_gripper_finger_link.get_position()[2]]))
        # Punish or reward?
        sign = 5 if distance < (self.max_ceiling - self.min_floor) else -1

        return abs(1 - distance / self.max_ceiling) * sign

    def execute_joint_lock(self, should_lock=None):
        if should_lock is not None and type(should_lock) is bool:
            self.joints_are_locked = should_lock

        if self.joints_are_locked:
            self.robot.lock_joints = [True] * self.action_space.shape[0]
            self.robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
            self.robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
            self.robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
            self.robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
        else:
            self.robot.lock_joints = [False] * self.action_space.shape[0]

        self.joints_are_locked = not self.joints_are_locked


class FetchInternalKeepStillTrainEnv(BaseFetchEnv, ABC):
    """
    The reward functions for this env will involve lifting the arm overhead

    """
    def __init__(self, is_sanity_test=False, distance_threshold = 0, reward_type='default', power=0.2):
        super().__init__(is_sanity_test, distance_threshold, reward_type, power)

        self.joints_not_at_limit_cost = .3
        self.init_positions = [0] * self.action_space.shape[0]
        self.init_positions[11] = -0.2
        self._index = 0

    def init_robot_pose(self):
        """ UPDATE ACTIONS """
        if not self.scene.multiplayer:
            # if multiplayer, action first applied to all robots, then global step() called, then _step()
            # for all robots with the same actions
            self.robot.apply_positions(self.init_positions, 0.9)
            self._index += 1
        self.scene.global_step()

        """ CALCULATE STATES """
        # also calculates self.joints_at_limit
        self.get_full_state()

        if self._index > 50:
            self._index = 0
            return True
        else:
            return False

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = PickAndMoveScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def get_custom_reward(self, achieved_goal=None, goal=None):
        """
        Reward Joints that are not breaking their limits.

        :return:
        """
        return float(self.joints_not_at_limit_cost * (len(self.robot.ordered_joints) - self.robot.joints_at_limit)) + \
               float(
                   self.joints_not_at_limit_cost * (len(self.robot.ordered_joints) - self.robot.joints_at_speed_limit))

    def execute_joint_lock(self, should_lock=None):
        if should_lock is not None and type(should_lock) is bool:
            self.joints_are_locked = should_lock

        if self.joints_are_locked:
            self.robot.lock_joints = [True] * self.action_space.shape[0]
            self.robot.lock_joints[10] = False  # Unlock 'shoulder_pan_joint
            self.robot.lock_joints[11] = False  # Unlock 'shoulder_lift_joint'
            self.robot.lock_joints[13] = False  # Unlock 'elbow_flex_joint'
            self.robot.lock_joints[15] = False  # Unlock 'wrist_flex_joint'
            self.robot.lock_joints[18] = False  # Unlock 'r_gripper_finger_joint'
            self.robot.lock_joints[19] = False  # Unlock 'l_gripper_finger_joint'
        else:
            self.robot.lock_joints = [False] * self.action_space.shape[0]

        self.joints_are_locked = not self.joints_are_locked
