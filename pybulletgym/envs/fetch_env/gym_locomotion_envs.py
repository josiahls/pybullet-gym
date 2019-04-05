import pybullet
from abc import ABC

import numpy as np
from pybullet_envs.bullet import bullet_client
from pybullet_envs.bullet.bullet_client import BulletClient

from .env_bases import BaseBulletEnv
from .robot_locomotors import FetchURDF
from .scene_manipulators import PickKnifeAndCutTestScene, PickAndMoveScene, KnifeCutScene, SceneFetch
from .scene_object_bases import TargetSceneObject, SlicableSceneObject, SlicingSceneObject


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

        self.joints_at_limit_cost = -0.1
        self.scene = None
        self.potential = 0
        self.rewards = []
        self.stateId = -1
        self.frame = 0
        self.done = 0
        self.reward = 0
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


class FetchPickKnifeAndCutTestEnv(BaseFetchEnv, ABC):

    def __init__(self):
        self.robot = FetchURDF()
        BaseBulletEnv.__init__(self, self.robot)

        self.joints_at_limit_cost = -0.1
        self.scene = None
        self.potential = 0
        self.rewards = []
        self.stateId = -1
        self._p = None

    def create_single_player_scene(self, _p: BulletClient):

        self.scene = PickKnifeAndCutTestScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.scene

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
        l_grasp_distance = 0  # np.linalg.norm(np.subtract(self.robot.l_gripper_finger_link.get_position(),
        # self.scene.objects_of_interest['knife.urdf'].get_position()))
        r_grasp_distance = 0  # np.linalg.norm(np.subtract(self.robot.r_gripper_finger_link.get_position(),
        # self.scene.objects_of_interest['knife.urdf'].get_position()))

        """ Distance of knife edge to target cube """
        knife_distance = 0  # np.linalg.norm(np.subtract(self.scene.objects_of_interest['knife.urdf'].get_position(),
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


class FetchMoveBlockEnv(BaseFetchEnv, ABC):

    def __init__(self):
        self.robot = FetchURDF()
        BaseBulletEnv.__init__(self, self.robot)

        self.joints_at_limit_cost = -0.1
        self.pick_and_place_scene = None
        self.potential = 0
        self.rewards = []
        self.elapsed_time = 0
        self.stateId = -1
        self.max_state_space_object_size = 30
        self._p = None

    def create_single_player_scene(self, _p: BulletClient):

        self.pick_and_place_scene = PickAndMoveScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.pick_and_place_scene

    def get_state_space_size(self):
        state = self.robot.calc_state().reshape((1, -1))
        object_states, distances = self.scene.calc_state()
        for object_state in object_states:
            state = np.hstack((state, np.array(object_state).reshape((1, -1))))
        for i in range(self.max_state_space_object_size - len(object_states)):
            state = np.hstack((state, np.zeros((1, len(list(self.scene.object_features.values()))))))

        return state.shape

    def step(self, a):

        """ UPDATE ACTIONS """
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        """ CALCULATE STATES """
        # also calculates self.joints_at_limit
        state = self.robot.calc_state()
        object_states, distances = self.scene.calc_state()

        """ CALCULATE REWARDS """
        # For no, the robot will always be alive
        # Otherwise, if the robot is upright, reward it
        alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[
            1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        # alive = 1
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True
            alive *= 10
        if done:
            print(f'Done because: state[0] is {state[0]} and the initial z is: {self.robot.initial_z}')
        print(f'Done because: state[0] is {state[0]} and the initial z is: {self.robot.initial_z}')

        # Punish higher amounts of time
        self.elapsed_time += 0.01

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
        object_positions = []
        for scene_object in list(reversed([_ for _ in self.scene.scene_objects if not _.removed])):
            if type(scene_object) is SlicableSceneObject:
                object_positions.append(scene_object.get_position())

        l_grasp_distance = np.linalg.norm(np.subtract(self.robot.l_gripper_finger_link.get_position(), object_positions), axis=1)
        r_grasp_distance = np.linalg.norm(np.subtract(self.robot.r_gripper_finger_link.get_position(), object_positions), axis=1)

        contact_events = [0]
        for scene_object in list(reversed([_ for _ in self.scene.scene_objects if not _.removed])):
            if type(scene_object) is TargetSceneObject:
                c1 = self._p.getContactPoints(self.robot.l_gripper_finger_link.bodyIndex, scene_object.bodyIndex,
                                              self.robot.l_gripper_finger_link.bodyPartIndex, scene_object.bodyPartIndex)
                c2 = self._p.getContactPoints(self.robot.r_gripper_finger_link.bodyIndex, scene_object.bodyIndex,
                                              self.robot.r_gripper_finger_link.bodyPartIndex, scene_object.bodyPartIndex)
                if c1 is not None:
                    contact_events.append(.1)
                if c2 is not None:
                    contact_events.append(.1)

        object_to_target_distances = []
        for scene_object in list(reversed([_ for _ in self.scene.scene_objects if not _.removed])):
            if type(scene_object) is TargetSceneObject:
                for position in object_positions:
                    object_to_target_distances.append(np.linalg.norm(scene_object.get_position() - position))

        """ Distance of knife edge to target cube """
        total_sum_target_distance = np.sum(distances)

        self.rewards = [
            alive,
            sum(contact_events),
            -1 * self.elapsed_time,
            # -1 * total_sum_target_distance / 20,
            # progress,
            # electricity_cost,
            -1 * sum([abs(_) > 1 for _ in a]),
            # sum([abs(_) < 1 for _ in a]),
            joints_at_limit_cost,
            # -1 * sum(l_grasp_distance) / 20,
            # -1 * sum(r_grasp_distance) / 20,
            -1 * sum(object_to_target_distances)
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        for object_state in object_states:
            state = np.hstack((state, object_state))

        return state, sum(self.rewards), bool(done), {}


class FetchCutBlockEnv_v1(BaseFetchEnv, ABC):

    def __init__(self):
        self.robot = FetchURDF()
        BaseBulletEnv.__init__(self, self.robot)

        self.joints_at_limit_cost = -0.1
        self.knife_cut_scene = None
        self.potential = 0
        self.rewards = []
        self.elapsed_time = 0
        self.stateId = -1
        self.max_state_space_object_size = 30
        self._p = None

    def create_single_player_scene(self, _p: BulletClient):

        self.knife_cut_scene = KnifeCutScene(_p, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.knife_cut_scene

    def get_state_space_size(self):
        state = self.robot.calc_state().reshape((1, -1))
        object_states, distances = self.scene.calc_state()
        for object_state in object_states:
            state = np.hstack((state, np.array(object_state).reshape((1, -1))))
        for i in range(self.max_state_space_object_size - len(object_states)):
            state = np.hstack((state, np.zeros((1, len(list(self.scene.object_features.values()))))))

        return state.shape

    def step(self, a):

        """ UPDATE ACTIONS """
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        """ CALCULATE STATES """
        # also calculates self.joints_at_limit
        state = self.robot.calc_state()
        object_states, distances = self.scene.calc_state()

        """ CALCULATE REWARDS """
        # For no, the robot will always be alive
        # Otherwise, if the robot is upright, reward it
        alive = float(self.robot.alive_bonus(state[0] - self.robot.initial_z, self.robot.body_rpy[
            1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        # alive = 1
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True
            alive *= 10
        if done:
            print(f'Done because: state[0] is {state[0]} and the initial z is: {self.robot.initial_z}')

        # Punish higher amounts of time
        self.elapsed_time += 0.01

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
        object_positions = []
        for scene_object in list(reversed([_ for _ in self.scene.scene_objects if not _.removed])):
            if type(scene_object) is SlicableSceneObject:
                object_positions.append(scene_object.get_position())

        l_grasp_distance = np.linalg.norm(np.subtract(self.robot.l_gripper_finger_link.get_position(), object_positions), axis=1)
        r_grasp_distance = np.linalg.norm(np.subtract(self.robot.r_gripper_finger_link.get_position(), object_positions), axis=1)

        contact_events = [0]
        for scene_object in list(reversed([_ for _ in self.scene.scene_objects if not _.removed])):
            if type(scene_object) is SlicingSceneObject:
                c1 = self._p.getContactPoints(self.robot.l_gripper_finger_link.bodyIndex, scene_object.bodyIndex,
                                              self.robot.l_gripper_finger_link.bodyPartIndex, scene_object.bodyPartIndex)
                c2 = self._p.getContactPoints(self.robot.r_gripper_finger_link.bodyIndex, scene_object.bodyIndex,
                                              self.robot.r_gripper_finger_link.bodyPartIndex, scene_object.bodyPartIndex)
                if c1 is not None:
                    contact_events.append(.1)
                if c2 is not None:
                    contact_events.append(.1)

        object_to_target_distances = []
        for scene_object in list(reversed([_ for _ in self.scene.scene_objects if not _.removed])):
            if type(scene_object) is SlicingSceneObject:
                for position in object_positions:
                    object_to_target_distances.append(np.linalg.norm(scene_object.get_position() - position))

        """ Distance of knife edge to target cube """
        total_sum_target_distance = np.sum(distances)

        self.rewards = [
            alive,
            sum(contact_events),
            len(self.scene.scene_objects),
            -1 * self.elapsed_time,
            # -1 * total_sum_target_distance / 20,
            # progress,
            # electricity_cost,
            -1 * sum([abs(_) > 1 for _ in a]),
            # sum([abs(_) < 1 for _ in a]),
            joints_at_limit_cost,
            # -1 * sum(l_grasp_distance) / 20,
            # -1 * sum(r_grasp_distance) / 20,
            -1 * sum(object_to_target_distances)
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        for object_state in object_states:
            state = np.hstack((state, object_state))

        done = True if sum(self.rewards) < -20 else done

        return state, sum(self.rewards), bool(done), {}