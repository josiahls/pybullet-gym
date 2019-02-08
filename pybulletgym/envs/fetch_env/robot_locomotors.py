import numpy as np
from pybullet_envs.bullet.bullet_client import BulletClient
from fetch_env.robot_bases import URDFBasedRobot

from pybulletgym.envs.mujoco.robot_bases import XmlBasedRobot, MJCFBasedRobot


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


class FetchURDF(MobileManipulatorBase, URDFBasedRobot):

    def __init__(self):
        MobileManipulatorBase.__init__(self, power=0.75)
        URDFBasedRobot.__init__(self, "fetch/fetch_description/robots/fetch.urdf", "base_link", action_dim=25,
                                obs_dim=70)

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


class FetchMJCF(MobileManipulatorBase, MJCFBasedRobot):

    def __init__(self):
        MobileManipulatorBase.__init__(self, power=0.75)
        MJCFBasedRobot.__init__(self, "fetch/main.xml", "base_link", action_dim=15,
                                obs_dim=30)

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
