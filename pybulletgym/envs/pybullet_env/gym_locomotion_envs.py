import numpy as np

from .scene_pick_and_place import PickAndPlaceScene
from .env_bases import BaseBulletEnv
from .robot_locomotors import FetchURDF, FetchMJCF


class FetchPickAndPlaceEnv(BaseBulletEnv):
    def __init__(self):
        self.robot = FetchMJCF()
        BaseBulletEnv.__init__(self, self.robot)

        self.joints_at_limit_cost = -0.1

    # def create_single_player_scene(self, bullet_client):
    #     self.stadium_scene = StadiumScene(bullet_client, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
    #     return self.stadium_scene

    def create_single_player_scene(self, bullet_client):
        self.pick_and_place_scene = PickAndPlaceScene(bullet_client, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.pick_and_place_scene

    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        # alive = float(self.robot.alive_bonus(state[0]+self.robot.initial_z, self.robot.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        alive = 1  # For no, the robot will always be alive
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        # electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        # electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

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

        self.rewards = [
            alive,
            progress,
            # electricity_cost,
            joints_at_limit_cost,
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}