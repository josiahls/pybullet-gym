from abc import ABC

import numpy as np
from pybullet_envs.bullet.bullet_client import BulletClient
import pybullet
from .scene_manipulators import PickKnifeAndCutScene
from .env_bases import BaseBulletEnv
from .robot_locomotors import FetchURDF
from pybullet_envs.bullet import bullet_client
from typing import List


class FetchPickKnifeAndCutEnv(BaseBulletEnv, ABC):

    def __init__(self):
        self.robot = FetchURDF()
        BaseBulletEnv.__init__(self, self.robot)

        self.joints_at_limit_cost = -0.1
        self.pick_and_place_scene = None
        self.potential = 0
        self.rewards = []
        self.stateId = -1
        self._p = None

    def create_single_player_scene(self, _p: BulletClient):

        self.pick_and_place_scene = PickKnifeAndCutScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.pick_and_place_scene

    def _reset(self):
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

        # self.robot.scene = self.scene

        self.frame = 0
        self.done = 0
        self.reward = 0
        dump = 0
        s = self.robot.reset(self._p, scene=self.scene)
        self.robot.robot_specific_reset(self._p)
        self.camera._p = self._p
        self.potential = self.robot.calc_potential(scene=self.scene)

        return s

    def reset(self):
        self._reset()

        if self.stateId >= 0:
            # print("restoreState self.stateId:",self.stateId)
            self._p.restoreState(self.stateId)

        if self.stateId < 0:
            self.stateId = self._p.saveState()

    def step(self, a):

        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        # also calculates self.joints_at_limit
        state = self.robot.calc_state()
        object_states = self.scene.calc_state()

        # For no, the robot will always be alive
        # Otherwise, if the robot is upright, reward it
        alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[
            1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        # alive = 1
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        # This is the closeness to the goal. this will be determined based on closeness to the knife
        potential_old = self.potential
        self.potential = self.robot.calc_potential(scene=self.scene)
        progress = float(self.potential - potential_old)

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if debugmode:
            print("alive=")
            print(alive)
            print("progress")
            print(progress)
            # print("electricity_cost")
            # print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)

        """ Grasp Reward (straight line distance) """
        l_grasp_distance = 0 # np.linalg.norm(np.subtract(self.robot.l_gripper_finger_link.get_position(),
                                                     # self.scene.objects_of_interest['knife.urdf'].get_position()))
        r_grasp_distance = 0 # np.linalg.norm(np.subtract(self.robot.r_gripper_finger_link.get_position(),
                                                     # self.scene.objects_of_interest['knife.urdf'].get_position()))

        """ Distance of knife edge to target cube """
        knife_distance = 0#np.linalg.norm(np.subtract(self.scene.objects_of_interest['knife.urdf'].get_position(),
                           #                         self.scene.objects_of_interest['cube_target.urdf'].get_position()))

        self.rewards = [
            # alive,
            # progress,
            # electricity_cost,
            joints_at_limit_cost,
            -1 * l_grasp_distance,
            -1 * r_grasp_distance
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}

    def camera_adjust(self):
        x, y, z = self.robot.body_xyz
        self.camera_x = 0.98 * self.camera_x + (1 - 0.98) * x
        self.camera.move_and_look_at(self.camera_x, y - 2.0, 1.4, x, y, 1.0)
