import numpy as np
from pybullet_envs.bullet.bullet_client import BulletClient
from pybullet_envs.scene_abstract import Scene
import os

from typing import List, Dict

from .robot_bases import BodyPart
from .utils import Normalizer
from .robot_bases import URDFBasedRobot
from .robot_bases import Joint
import pybullet
from pybullet_envs.bullet import bullet_client

"""
# orientation = self._p.getEulerFromQuaternion(self.parts['gripper_link'].get_orientation())
# init_orientation = self._p.getEulerFromQuaternion(self.parts['gripper_link'].initialOrientation)
# normal = np.array(orientation) @ np.array(init_orientation)
# print(f'Normal {np.array(orientation) @ np.array(init_orientation)} Diff {orientation} Pos: {pos}')

"""

from pybulletgym.envs.mujoco.robot_bases import XmlBasedRobot, MJCFBasedRobot


class FetchURDF(URDFBasedRobot):

    def __init__(self, power=0.2, action_locks=None):
        URDFBasedRobot.__init__(self, "fetch/fetch_description/robots/fetch.urdf", "base_link", action_dim=25,
                                obs_dim=70, self_collision=True)
        self.power = power
        self.orthogonal_wrist = False
        self.initial_wrist_orientation = None
        # Keeps the finger locations on their closed position
        self.block_gripper = False
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0.01
        self.manipulator_target_x = 1e3  # kilometers away
        self.manipulator_target_y = 0
        self.body_xyz = [0, 0, 0]
        self.body_rpy = [0, 0, 0]
        self.initial_z = None
        self.joint_speeds = []
        self.joints_at_limit = []
        self.normalize_state_space_output = True
        self.parts = None  # type: Dict[str, BodyPart]
        self.jdict = None  # type: Dict[str, Joint]
        self.ordered_joints = None  # type: Dict[Joint]
        self.robot_body = None
        self.action_space_only_unlocked = False
        if action_locks is None:
            self.lock_joints = [False] * self.action_space.shape[0]
        else:
            self.lock_joints = action_locks

        self.pos_after = 0

    def calc_state(self):
        """
        Calculates the state of the joints in the robot, and its position

        :return:
        """

        """ Get joint information, what joints are at their limits """
        qpos = np.array([j.get_position() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (25,)
        qvel = np.array([j.get_velocity() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (25,)

        temp_var = [str(j.current_relative_position()) + ' ' + j.joint_name for j in self.ordered_joints]
        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_speed_limit = np.count_nonzero([j1.jointMaxVelocity < np.abs(self.joint_speeds[i]) for i, j1 in enumerate(self.ordered_joints)])
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 1)

        """ Set the robot's general body position. Primarily concerned with torso (base position). """
        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (
            parts_xyz[0::3].mean(), parts_xyz[1::3].mean(),
            body_pose.xyz()[2])  # torso z is more informative than mean z
        self.body_rpy = body_pose.rpy()

        """ 
            Update z, and especially initial z. Unless the robot is supposed to jump, its z really should
            never change...
        """
        z = self.body_xyz[2]
        if self.initial_z is None:
            self.initial_z = z

        state = np.concatenate([
            qpos.flat[1:],  # self.sim.data.qpos.flat[1:],
            np.clip(qvel, -1, 1).flat  # self.sim.data.qvel.flat,
        ])

        if self.normalize_state_space_output:
            state = Normalizer.normalize(state.reshape(1, -1), f'fetch_robot_{state.shape}')

        return state

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        i = 0
        ii = 0
        for n, j in enumerate(self.ordered_joints):
            if j.power_coef != 0 and not self.lock_joints[n]:  # in case the ignored joints are added, they have 0 power
                _a = a[n - i] if not self.action_space_only_unlocked else a[ii]
                ii += 1
                j.set_motor_torque(self.power * j.power_coef * float(np.clip(_a, -1, +1)))
            elif self.lock_joints[n]:
                j.set_velocity(0)
            else:
                i += 1

    def apply_positions(self, a, maxVelocity=None):
        assert (np.isfinite(a).all())
        i = 0
        ii = 0
        for n, j in enumerate(self.ordered_joints):
            if j.power_coef != 0:  # in case the ignored joints are added, they have 0 power
                pos = j.get_position()
                if not self.lock_joints[n]:
                    _a = a[n - i] if not self.action_space_only_unlocked else a[ii]
                    ii += 1
                    j.set_position(pos + _a, maxVelocity)
                else:
                    if self.block_gripper and j.joint_name.__contains__('l_gripper_finger_joint'):
                        j.set_position(0, maxVelocity)
                    elif self.block_gripper and j.joint_name.__contains__('r_gripper_finger_joint'):
                        j.set_position(0, maxVelocity)
                    elif self.orthogonal_wrist and j.joint_name.__contains__('wrist_flex_joint'):
                        gripper_position = np.array(self.parts['gripper_link'].get_position())
                        wrist_position = np.array(self.parts['wrist_flex_link'].get_position())
                        gripper_orientation = np.array(self._p.getEulerFromQuaternion(self.parts['gripper_link'].get_orientation()))
                        wrist_orientation = np.array(self._p.getEulerFromQuaternion(self.parts['wrist_flex_link'].get_orientation()))
                        if self.initial_wrist_orientation is None:
                            self.initial_wrist_orientation = wrist_orientation

                        # Note, -1 direction curls in, +1 curls out
                        orientation_difference = abs(self.initial_wrist_orientation[1] - wrist_orientation[1])
                        position_difference = gripper_position[0] - wrist_position[0]
                        j.set_position(pos - np.sign(position_difference) * orientation_difference, 0.5)
                    else:
                        j.set_position(pos, maxVelocity)
            else:
                i += 1

    def reset(self, bullet_client, **kwargs):
        self._p = bullet_client
        # self.ordered_joints = []
        full_path = os.path.join(os.path.dirname(__file__), "..", "assets", "robots", self.model_urdf)

        if self.self_collision and self.robot_body is None:
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p,
                                                                                           self._p.loadURDF(full_path,
                                                                                                            basePosition=self.basePosition,
                                                                                                            baseOrientation=self.baseOrientation,
                                                                                                            useFixedBase=self.fixed_base,
                                                                                                            flags=pybullet.URDF_USE_SELF_COLLISION  | pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL | pybullet.URDF_USE_MATERIAL_TRANSPARANCY_FROM_MTL))
        elif self.robot_body is None:
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p,
                                                                                           self._p.loadURDF(full_path,
                                                                                                            basePosition=self.basePosition,
                                                                                                            baseOrientation=self.baseOrientation,
                                                                                                            useFixedBase=self.fixed_base,
                                                                                                            flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL | pybullet.URDF_USE_MATERIAL_TRANSPARANCY_FROM_MTL))

        self.robot_specific_reset(self._p)

        s = self.calc_state()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        self.potential = self.calc_potential(scene=kwargs['scene'])

        return s

    def calc_potential(self, **kwargs):
        scene = kwargs['scene']
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will
        # change 2-3 per frame (not per second), all rewards have rew/frame units and close to 1.0
        pos_before = self.pos_after
        self.pos_after = self.robot_body.get_pose()[0]
        debugmode = 0
        if debugmode:
            print("self.scene.dt")
            print(scene.dt)
            print("self.scene.frame_skip")
            print(scene.frame_skip)
            print("self.scene.timestep")
            print(scene.timestep)
        return (self.pos_after - pos_before) / scene.dt

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client

        for part in self.parts:
            self.parts[part].reset_pose(self.parts[part].initialPosition, self.parts[part].initialOrientation)
        self.reset_pose([0, 0, 0.01], [0, 0, 0, 1])
        self.initial_z = None

    def alive_bonus(self, z, pitch):
        # print(f'alive_bonus: {z} and {self.body_xyz[2]} subtraction is: {z - self.body_xyz[2]}')
        return 2 if abs(z - self.body_xyz[2]) < 0.05 else -4
        # return +2 if 2 > z > -0.2 and .28 > pitch > -.1 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying


class FetchMJCF(MJCFBasedRobot):

    def __init__(self, power=.75):
        MJCFBasedRobot.__init__(self, "fetch/main.xml", "base_link", action_dim=15,
                                obs_dim=30)
        self.power = power
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.manipulator_target_x = 1e3  # kilometers away
        self.manipulator_target_y = 0
        self.body_xyz = [0, 0, 0]
        self.pos_after = 0

    def calc_state(self):
        qpos = np.array([j.get_position() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (25,)
        qvel = np.array([j.get_velocity() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (25,)

        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        return np.concatenate([
            qpos.flat[1:],  # self.sim.data.qpos.flat[1:],
            np.clip(qvel, -10, 10).flat  # self.sim.data.qvel.flat,
        ])

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        i = 0
        for n, j in enumerate(self.ordered_joints):
            if j.power_coef != 0:  # in case the ignored joints are added, they have 0 power
                j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n - i], -1, +1)))
            else:
                i += 1

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        pos_before = self.pos_after
        self.pos_after = self.robot_body.get_pose()[0]
        debugmode = 0
        if debugmode:
            print("calc_potential: self.walk_target_dist")
            print(self.walk_target_dist)
            print("self.scene.dt")
            print(self.scene.dt)
            print("self.scene.frame_skip")
            print(self.scene.frame_skip)
            print("self.scene.timestep")
            print(self.scene.timestep)
        return (self.pos_after - pos_before) / self.scene.dt

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.scene.actor_introduce(self)
        for part in self.parts:
            self.parts[part].reset_pose(self.parts[part].initialPosition, self.parts[part].initialOrientation)
        # self.reset_pose([0, 0, 0], [0, 0, 0, 1])
        self.initial_z = None

    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying
