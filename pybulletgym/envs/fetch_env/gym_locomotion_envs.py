import pybullet
import random
from abc import ABC

import numpy as np
from pybullet_envs.bullet import bullet_client
from pybullet_envs.bullet.bullet_client import BulletClient

from .env_bases import BaseBulletEnv
from .robot_locomotors import FetchURDF
from .scene_manipulators import PickKnifeAndCutTestScene, PickAndMoveScene, KnifeCutScene, SceneFetch
from .scene_object_bases import Features


class BaseFetchEnv(BaseBulletEnv, ABC):
    def __init__(self):
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
        self.robot = FetchURDF()
        BaseBulletEnv.__init__(self, self.robot)

        self.joints_at_limit_cost = -1.
        self.scene = None
        self.potential = 0
        self.rewards = []
        self.stateId = -1
        self.frame = 0
        self.done = 0
        self.reward = 0
        self.elapsed_time = 0
        self.max_state_space_object_size = 3
        self._p = None

    def reset(self):
        """
        More Fetch specific reset functionality.

        One of the biggest differences is removing adding a scene to the robot.
        The goal here is to reduce the number of instances of a scene object to 1.

        :return:
        """
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
        self.reward = 0
        self.elapsed_time = 0
        s = self.robot.reset(self._p, scene=self.scene)
        self.robot.robot_specific_reset(self._p)
        self.camera._p = self._p
        self.potential = self.robot.calc_potential(scene=self.scene)

        # Before adding dynamic objects, we save the original state
        if self.stateId < 0:
            self.stateId = self._p.saveState()

        # Load dynamic objects
        self.scene.dynamic_object_load(self._p)

        return s

    def create_single_player_scene(self, _p: BulletClient):
        return SceneFetch(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)

    def camera_adjust(self):
        x, y, z = self.robot.body_xyz
        self.camera_x = 0.98 * self.camera_x + (1 - 0.98) * x
        self.camera.move_and_look_at(self.camera_x, y - 2.0, 1.4, x, y, 1.0)

    def get_full_state(self) -> np.array:
        """
        Returns a singular state space 1D representation of the environment's space.

        The base env has a fixed limit of number of objects to track, and so if the number
        of objects < max_state_space_object_size then the remaining space is filled with zeros

        :return:
        """
        # Get the state of the robot
        state = self.robot.calc_state().reshape((1, -1))
        object_states = self.scene.calc_state()
        for i, object_state in enumerate(object_states):
            if i < self.max_state_space_object_size:
                state = np.hstack((state, np.array(object_state).reshape((1, -1))))

        for i in range(self.max_state_space_object_size - len(object_states)):
            state = np.hstack((state, np.zeros((1, len(Features())))))

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
        state = self.get_full_state()

        """ CALCULATE REWARDS (internal to the robot)"""
        # For no, the robot will always be alive
        # Otherwise, if the robot is upright, reward it
        # alive = float(self.robot.alive_bonus(state[0][0] + self.robot.initial_z, self.robot.body_rpy))
        alive = float(self.robot.alive_bonus(self.robot.initial_z, self.robot.body_rpy))
        # alive = 1
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True
        if done:
            print(f'Done because: state[0] is {state[0][0]} and the initial z is: {self.robot.initial_z} and the rpy '
                  f'rpy is {self.robot.body_rpy} rxy is {self.robot.body_xyz}')

        # Punish higher amounts of time
        # self.elapsed_time += 0.01

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
        if abs(custom_reward) > 1:
            print('WARNING REWARD IS NOT NORMALIZED')

        self.rewards = [
            alive,
            -1 * self.elapsed_time,
            joints_at_limit_cost,
            joints_at_speed_limit_cost,
            custom_reward,
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)

        return state, sum(self.rewards), bool(done), {}

    def get_custom_reward(self):
        return 0


class FetchPickKnifeAndCutTestEnv(BaseFetchEnv, ABC):

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = PickKnifeAndCutTestScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene


class FetchMoveBlockEnv(BaseFetchEnv, ABC):

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = PickAndMoveScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene


class FetchCutBlockEnv_v1(BaseFetchEnv, ABC):

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = KnifeCutScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene


class FetchInternalTrainEnv(BaseFetchEnv, ABC):
    """
    The reward functions for this env will involve lifting the arm overhead

    """
    def __init__(self):
        super().__init__()

        self.randomCeiling = random.uniform(.5, 2)

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = PickAndMoveScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def get_custom_reward(self):
        return (self.robot.l_gripper_finger_link.get_position()[2] - self.randomCeiling) / self.randomCeiling or \
               (self.robot.r_gripper_finger_link.get_position()[2] - self.randomCeiling) / self.randomCeiling

class FetchInternalKeepStillTrainEnv(BaseFetchEnv, ABC):
    """
    The reward functions for this env will involve lifting the arm overhead

    """
    def __init__(self):
        super().__init__()

        self.joints_not_at_limit_cost = .3
        self.robot.lock_joints = [True] * self.action_space.shape[0]

    def create_single_player_scene(self, _p: BulletClient):
        self.scene = PickAndMoveScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

    def get_custom_reward(self):
        """
        Reward Joints that are not breaking their limits.

        :return:
        """
        return float(self.joints_not_at_limit_cost * (len(self.robot.ordered_joints) - self.robot.joints_at_limit)) + \
               float(self.joints_not_at_limit_cost * (len(self.robot.ordered_joints) - self.robot.joints_at_speed_limit))
