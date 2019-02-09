import inspect
import os
import pybullet

from .scene_bases import Scene

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)


class PickAndPlaceScene(Scene):
    """
    The goal of this scene is to set up a scene for picking up and moving
    an object to another location.

    """
    multiplayer = False
    sceneLoaded = 0

    def episode_restart(self, bullet_client: pybullet):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)
        if self.sceneLoaded == 0:
            self.sceneLoaded = 1

            # Load the table
            filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "table",
                                    "table.urdf")
            self._p.loadURDF(filename, [1, 0, 0], [0, 0, 90, 90])
            # Load the plane
            filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "plane",
                                    "plane.urdf")
            self._p.loadURDF(filename)

            # Load the cube
            filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "cubes",
                                    "cube_target_no_collision.urdf")
            self._p.loadURDF(filename, [1, -0.3, 0.75])

            # Load the cube
            filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "cubes",
                                    "cube_small.urdf")
            self._p.loadURDF(filename, [1, 0.3, 0.65])


class PickKnifeAndCutScene(Scene):
    """
    The goal of this scene is to set up a scene for picking up a knife, and cutting a sphere or a square

    """
    multiplayer = False
    sceneLoaded = 0

    def episode_restart(self, bullet_client: pybullet):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)
        if self.sceneLoaded == 0:
            self.sceneLoaded = 1

            filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "table",
                                    "table.urdf")
            self._p.loadURDF(filename, [1, 0, 0], [0, 0, 90, 90])
            # Load the plane
            filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "plane",
                                    "plane.urdf")
            self._p.loadURDF(filename)

            # Load the cube
            filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "cubes",
                                    "cube_target_no_collision.urdf")
            self._p.loadURDF(filename, [0.8, -0.4, 0.70])

            # Load the cube
            filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "food", 'orange',
                                    "orange.urdf")
            self._p.loadURDF(filename, [0.8, 0.4, 0.70],
                             flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL |
                                   pybullet.URDF_USE_MATERIAL_TRANSPARANCY_FROM_MTL)

            # # Load the knife
            filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "knives",
                                    "knife.urdf")
            self._p.loadURDF(filename, [0.8, 0.2, 0.65], self._p.getQuaternionFromEuler([90, 0, 80]),
                             flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL |
                                   pybullet.URDF_USE_MATERIAL_TRANSPARANCY_FROM_MTL)
