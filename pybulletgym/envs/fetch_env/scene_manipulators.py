import inspect
import os
import pybullet
from typing import List, Dict

from pybullet_envs.bullet.bullet_client import BulletClient

from robot_bases import BodyPart
from .scene_bases import Scene

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)


class PickAndPlaceScene(Scene):
    """
    The goal of this scene is to set up a scene for picking up and moving
    an object to another location.

    """
    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        super().__init__(bullet_client, gravity, timestep, frame_skip)

        self.multiplayer = False
        self.sceneLoaded = 0
        self.objects_of_interest = []  # type: List[str]

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
    def __init__(self, objects_of_interest: List[str], bullet_client, gravity, timestep, frame_skip):
        super().__init__(bullet_client, gravity, timestep, frame_skip)

        self.multiplayer = False
        self.sceneLoaded = 0
        self._objects_to_load = objects_of_interest  # type: List[str]
        self.objects_of_interest = {}  # type: Dict[BodyPart]

    def episode_restart(self, bullet_client: pybullet):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)

        """ If the scene isn't loaded, then load the models """
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

            filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "knives",
                                    "knife.urdf")
            self._p.loadURDF(filename, [0.75, 0.3, 1.2], self._p.getQuaternionFromEuler([90, 0, 80]),
                             flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL |
                                   pybullet.URDF_USE_MATERIAL_TRANSPARANCY_FROM_MTL)

    def dynamic_object_episode_restart(self, bullet_client: BulletClient):
        self._p = bullet_client

        # If the original cube was removed, reload
        if 'cube_concave.urdf' not in self.objects_of_interest:
            filename = os.path.join(os.path.dirname(__file__), "..", "assets", "things", "cubes",
                                    "cube_concave.urdf")
            self._p.loadURDF(filename, [0.8, 0.3, 0.70])

        """ Load the interfaces for the objects of interest """
        for i in range(self._p.getNumBodies()):
            # If a body is an object of interest
            if self._p.getBodyInfo(i)[1].decode("utf-8") in self._objects_to_load:
                print(f'Found {self._p.getBodyInfo(i)[1]}')
                base_link_name, object_name = self._p.getBodyInfo(i)
                object_name = object_name.decode("utf8")
                base_link_name = base_link_name.decode("utf8")
                self.objects_of_interest[object_name] = BodyPart(self._p, base_link_name,
                                                                 list(range(self._p.getNumBodies())),
                                                                 i, -1)
                for j in range(self._p.getNumJoints(i)):
                    joint_info = self._p.getJointInfo(i, j)
                    part_name = joint_info[12].decode("utf8")
                    self.objects_of_interest[part_name] = BodyPart(self._p, part_name,
                                                                   list(range(self._p.getNumBodies())), i, j)

    def calc_state(self):
        """
        We want to update the states of the objects of interest during manipulation.
        The main interest is the position of each of the objects of interest so we can
        easily build a reward system around them.

        We also want to be able to add advanced object behaviors such as breaking into smaller
        pieces.

        :return:
        """

        """ Handle the knife blade collision """
        if 'cube_concave.urdf' not in self.objects_of_interest or 'blade' not in self.objects_of_interest:
            print('One of these is missing! Skipping !!!')
            return 0

        collision = self._p.getOverlappingObjects(self.objects_of_interest['blade'].get_position(),
                                            self.objects_of_interest['cube_concave.urdf'].get_position())

        # If a collision is detected in this region, then start checking for cutting
        if collision is not None:
            # Get the unique ids (body ids and parts ids) for both bodies.
            blade_id = (self.objects_of_interest['blade'].bodyIndex, self.objects_of_interest['blade'].bodyPartIndex)
            cube_concave_id = (self.objects_of_interest['cube_concave.urdf'].bodyIndex,
                               self.objects_of_interest['cube_concave.urdf'].bodyPartIndex)

            # If they are both in the list then cutting is happening
            if blade_id in collision and cube_concave_id in collision:
                # Allow the knife to pass through the cube
                self._p.setCollisionFilterPair(blade_id[0], cube_concave_id[0], blade_id[1], cube_concave_id[1], False)

                # Get the cube data:
                cube_data = self._p.getCollisionShapeData(cube_concave_id[0], cube_concave_id[1])[0]
                height, width, length = cube_data[3]
                position = self.objects_of_interest['cube_concave.urdf'].get_position()
                orientation = self.objects_of_interest['cube_concave.urdf'].get_orientation()
                # Get the knife data
                knife_position = self.objects_of_interest['knife.urdf'].get_position()
                knife_orientation = self.objects_of_interest['knife.urdf'].get_orientation()

                # Removing the cube
                self._p.removeBody(cube_concave_id[0])
                self.objects_of_interest.pop('cube_concave.urdf')

                # Add the slices instead
                # Create the collision shapes
                collision_slice_1 = self._p.createCollisionShape(self._p.GEOM_BOX, halfExtents=[.1,.05,.1])
                collision_slice_2 = self._p.createCollisionShape(self._p.GEOM_BOX, halfExtents=[.1,.05,.1])
                visual_shape_slice_1 = self._p.createVisualShape(self._p.GEOM_BOX, halfExtents=[.1,.05,.1], rgbaColor=[1, 0, 0, 1])
                visual_shape_slice_2 = self._p.createVisualShape(self._p.GEOM_BOX, halfExtents=[.1,.05,.1], rgbaColor=[1, 0, 0, 1])

                # orientation (z location ) = the knife z location
                self._p.createMultiBody(collision_slice_1, visual_shape_slice_1, basePosition=position,
                                        baseOrientation=orientation)
                self._p.createMultiBody(collision_slice_2, visual_shape_slice_2, basePosition=position,
                                        baseOrientation=orientation)



                print('cutting the cube!!!!')
            else:
                print('not cutting teh cube')

        return 0
