from env_bases import BaseBulletEnv
from pybulletgym.envs.mujoco.robot_bases import XmlBasedRobot, MJCFBasedRobot, URDFBasedRobot
import numpy as np
from pybulletgym.envs import gym_utils as ObjectHelper
from pybullet_envs.bullet.bullet_client import BulletClient
import pybullet as p

from roboschool.robot_locomotors import WalkerBase


class MobileManipulatorBase(XmlBasedRobot):

    def __init__(self, power):
        self.power = power
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.manipulator_target_x = 1e3  # kilometers away
        self.manipulator_target_y = 0
        self.body_xyz = [0, 0, 0]

    def robot_specific_reset(self, bullet_client: BulletClient):
        self._p = bullet_client
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)

        self.scene.actor_introduce(self)
        self.initial_z = None

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        i = 0
        for n, j in enumerate(self.ordered_joints):
            if j.power_coef != 0:  # in case the ignored joints are added, they have 0 power
                j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n - i], -1, +1)))
            else:
                i += 1

    def calc_state(self):
        qpos = np.array([j.get_position() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (6,)
        qvel = np.array([j.get_velocity() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (6,)

        return np.concatenate([
            qpos.flat[1:],  # self.sim.data.qpos.flat[1:],
            np.clip(qvel, -10, 10).flat  # self.sim.data.qvel.flat,
        ])

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
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
        return - self.walk_target_dist / self.scene.dt


class Fetch(MobileManipulatorBase, MJCFBasedRobot):

    def __init__(self):
        MobileManipulatorBase.__init__(self, power=0.75)
        MJCFBasedRobot.__init__(self, "fetch/main.xml", "base_link", action_dim=15, obs_dim=11,
                                add_ignored_joints=True)

        self.pos_after = 0

    def calc_state(self):
        qpos = np.array([j.get_position() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (6,)
        qvel = np.array([j.get_velocity() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (6,)

        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        return np.concatenate([
            qpos.flat[1:],  # self.sim.data.qpos.flat[1:],
            np.clip(qvel, -10, 10).flat  # self.sim.data.qvel.flat,
        ])

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
        self.initial_z = None

    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying


class WalkerBase(XmlBasedRobot):
    def __init__(self, power):
        self.power = power
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.walk_target_x = 1e3  # kilometers away
        self.walk_target_y = 0
        self.body_xyz = [0, 0, 0]

    def robot_specific_reset(self, bullet_client: BulletClient):
        self._p = bullet_client
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)

        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
        self.scene.actor_introduce(self)
        self.initial_z = None

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        i = 0
        for n, j in enumerate(self.ordered_joints):
            if j.power_coef != 0:  # in case the ignored joints are added, they have 0 power
                j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n - i], -1, +1)))
            else:
                i += 1

    def calc_state(self):
        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (
            parts_xyz[0::3].mean(), parts_xyz[1::3].mean(),
            body_pose.xyz()[2])  # torso z is more informative than mean z
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        if self.initial_z is None:
            self.initial_z = z
        r, p, yaw = self.body_rpy
        self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                            self.walk_target_x - self.body_xyz[0])
        self.walk_target_dist = np.linalg.norm(
            [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
        angle_to_target = self.walk_target_theta - yaw

        rot_speed = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw), np.cos(-yaw), 0],
             [0, 0, 1]]
        )
        vx, vy, vz = np.dot(rot_speed, self.robot_body.speed())  # rotate speed back to body point of view

        more = np.array([z - self.initial_z,
                         np.sin(angle_to_target), np.cos(angle_to_target),
                         0.3 * vx, 0.3 * vy, 0.3 * vz,
                         # 0.3 is just scaling typical speed into -1..+1, no physical sense here
                         r, p], dtype=np.float32)
        return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
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
        return - self.walk_target_dist / self.scene.dt


class Hopper(WalkerBase, MJCFBasedRobot):
    """
	Hopper implementation based on MuJoCo.
	"""
    foot_list = ["foot"]

    def __init__(self):
        WalkerBase.__init__(self, power=0.75)
        MJCFBasedRobot.__init__(self, "hopper.xml", "torso", action_dim=3, obs_dim=11, add_ignored_joints=True)

        self.pos_after = 0

    def calc_state(self):
        qpos = np.array([j.get_position() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (6,)
        qvel = np.array([j.get_velocity() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (6,)

        return np.concatenate([
            qpos.flat[1:],  # self.sim.data.qpos.flat[1:],
            np.clip(qvel, -10, 10).flat  # self.sim.data.qvel.flat,
        ])

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


class Walker2D(WalkerBase, MJCFBasedRobot):
    """
	Walker2D implementation based on MuJoCo.
	"""
    foot_list = ["foot", "foot_left"]

    def __init__(self):
        WalkerBase.__init__(self, power=0.40)
        MJCFBasedRobot.__init__(self, "walker2d.xml", "torso", action_dim=6, obs_dim=17, add_ignored_joints=True)

        self.pos_after = 0

    def calc_state(self):
        qpos = np.array([j.get_position() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (9,)
        qvel = np.array([j.get_velocity() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (9,)

        return np.concatenate([
            qpos[1:],  # qpos[1:]
            np.clip(qvel, -10, 10)  # np.clip(qvel, -10, 10)
        ])

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
        WalkerBase.robot_specific_reset(self, bullet_client)
        for n in ["foot_joint", "foot_left_joint"]:
            self.jdict[n].power_coef = 30.0


class HalfCheetah(WalkerBase, MJCFBasedRobot):
    """
	Half Cheetah implementation based on MuJoCo.
	"""
    foot_list = ["ffoot", "fshin", "fthigh", "bfoot", "bshin", "bthigh"]  # track these contacts with ground

    def __init__(self):
        WalkerBase.__init__(self, power=1)
        MJCFBasedRobot.__init__(self, "half_cheetah.xml", "torso", action_dim=6, obs_dim=17, add_ignored_joints=True)

        self.pos_after = 0

    def calc_state(self):
        qpos = np.array([j.get_position() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (9,)
        qvel = np.array([j.get_velocity() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (9,)

        return np.concatenate([
            qpos.flat[1:],  # self.sim.data.qpos.flat[1:],
            qvel.flat  # self.sim.data.qvel.flat,
        ])

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
        WalkerBase.robot_specific_reset(self, bullet_client)
        for part_id, part in self.parts.items():
            self._p.changeDynamics(part.bodyIndex, part.bodyPartIndex, lateralFriction=0.8, spinningFriction=0.1,
                                   rollingFriction=0.1, restitution=0.5)

        self.jdict["bthigh"].power_coef = 120.0
        self.jdict["bshin"].power_coef = 90.0
        self.jdict["bfoot"].power_coef = 60.0
        self.jdict["fthigh"].power_coef = 140.0
        self.jdict["fshin"].power_coef = 60.0
        self.jdict["ffoot"].power_coef = 30.0


class Ant(WalkerBase, MJCFBasedRobot):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self):
        WalkerBase.__init__(self, power=2.5)
        MJCFBasedRobot.__init__(self, "ant.xml", "torso", action_dim=8, obs_dim=111)

    def calc_state(self):
        WalkerBase.calc_state(self)
        pose = self.parts['torso'].get_pose()
        qpos = np.hstack((pose, [j.get_position() for j in self.ordered_joints])).flatten()  # shape (15,)

        velocity = self.parts['torso'].get_velocity()
        qvel = np.hstack(
            (velocity[0], velocity[1], [j.get_velocity() for j in self.ordered_joints])).flatten()  # shape (14,)

        cfrc_ext = np.zeros((14, 6))  # shape (14, 6)  # TODO: FIND cfrc_ext
        return np.concatenate([
            qpos.flat[2:],  # self.sim.data.qpos.flat[2:],
            qvel.flat,  # self.sim.data.qvel.flat,
            np.clip(cfrc_ext, -1, 1).flat  # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground


class Humanoid(WalkerBase, MJCFBasedRobot):
    self_collision = True
    foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"

    def __init__(self, random_yaw=False, random_lean=False):
        WalkerBase.__init__(self, power=0.41)
        MJCFBasedRobot.__init__(self, 'humanoid_symmetric.xml', 'torso', action_dim=17, obs_dim=376)
        # 17 joints, 4 of them important for walking (hip, knee), others may as well be turned off, 17/4 = 4.25
        self.random_yaw = random_yaw
        self.random_lean = random_lean

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        self.motor_names = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power = [100, 100, 100]
        self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]
        if self.random_yaw:
            position = [0, 0, 0]
            orientation = [0, 0, 0]
            yaw = self.np_random.uniform(low=-3.14, high=3.14)
            if self.random_lean and self.np_random.randint(2) == 0:
                if self.np_random.randint(2) == 0:
                    pitch = np.pi / 2
                    position = [0, 0, 0.45]
                else:
                    pitch = np.pi * 3 / 2
                    position = [0, 0, 0.25]
                roll = 0
                orientation = [roll, pitch, yaw]
            else:
                position = [0, 0, 1.4]
                orientation = [0, 0, yaw]  # just face random direction, but stay straight otherwise
            self.robot_body.reset_position(position)
            self.robot_body.reset_orientation(p.getQuaternionFromEuler(orientation))
        self.initial_z = 0.8

    def calc_state(self):
        WalkerBase.calc_state(self)

        pose = self.parts['torso'].get_pose()
        qpos = np.hstack((pose, [j.get_position() for j in self.ordered_joints])).flatten()  # shape (24,)

        velocity = self.parts['torso'].get_velocity()
        qvel = np.hstack(
            (velocity[0], velocity[1], [j.get_velocity() for j in self.ordered_joints])).flatten()  # shape (23,)

        cinert = np.zeros((14, 10))  # shape (14, 10)  # TODO: FIND
        cvel = np.zeros((14, 6))  # shape (14, 6)  # TODO: FIND
        qfrc_actuator = np.zeros(23)  # shape (23,)  # TODO: FIND
        cfrc_ext = np.zeros((14, 6))  # shape (14, 6)  # TODO: FIND cfrc_ext
        return np.concatenate([
            qpos.flat[2:],  # self.sim.data.qpos.flat[2:],
            qvel.flat,  # self.sim.data.qvel.flat,
            cinert.flat,  # data.cinert.flat,
            cvel.flat,  # data.cvel.flat,
            qfrc_actuator.flat,  # data.qfrc_actuator.flat,
            cfrc_ext.flat  # data.cfrc_ext.flat
        ])

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        force_gain = 1
        for i, m, power in zip(range(17), self.motors, self.motor_power):
            m.set_motor_torque(float(force_gain * power * self.power * np.clip(a[i], -1, +1)))

    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying
