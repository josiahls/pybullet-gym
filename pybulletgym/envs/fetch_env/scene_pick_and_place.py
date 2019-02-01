import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from .scene_bases import Scene
import pybullet


class PickAndPlaceScene(Scene):
    multiplayer = False
    sceneLoaded = 0

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)
        if self.sceneLoaded == 0:
            self.sceneLoaded = 1

            # Load the table
            # filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "table_square",
            #                         "table_square.urdf")
            # self._p.loadURDF(filename, [-1, 0, 0])
            filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "table",
                                    "table.urdf")
            self._p.loadURDF(filename, [1, 0, 0], [0, 0, 90, 90])
            # Load the plane
            filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "plane",
                                    "plane.urdf")
            self._p.loadURDF(filename)

            # Load the cube
            filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "cubes",
                                    "cube_small.urdf")
            self._p.loadURDF(filename, [1, 0.3, 0.65])

            # # Load the sphere 1
            # filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "spheres",
            #                         "sphere2red.urdf")
            # self._p.loadURDF(filename, [1, -0.3, 5.8])

            # # Load the cube
            # filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "spheres",
            #                         "sphere_small_zeroinertia.urdf")
            # self._p.loadURDF(filename, [1, -0.3, 0.8])

