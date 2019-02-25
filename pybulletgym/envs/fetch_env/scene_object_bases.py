from typing import List

from pybullet_utils.bullet_client import BulletClient
import numpy as np
from robot_bases import BodyPart


class SceneObject(BodyPart):
    children: List[BodyPart]

    def __init__(self, bullet_client: BulletClient, filename: str = None, position: list = None, orientation: list = None,
                 flags=0, removable=False, reloadable=True):
        self.filename = filename
        if self.filename is not None:
            # Load the object mesh
            self.params = {'fileName': filename, 'basePosition': position, 'baseOrientation': orientation, 'flags': flags}
            self.params = {_: self.params[_] for _ in self.params if self.params[_] is not None}
            self.bodyIndex = bullet_client.loadURDF(**self.params)

        # Init the super object
        base_link_name, object_name = bullet_client.getBodyInfo(self.bodyIndex)
        base_link_name = base_link_name.decode("utf8")

        # Init the child joints
        self.children = []  # type: list
        for j in range(bullet_client.getNumJoints(self.bodyIndex)):
            joint_info = bullet_client.getJointInfo(self.bodyIndex, j)
            part_name = joint_info[12].decode("utf8")
            self.children.append(BodyPart(bullet_client, part_name, list(range(bullet_client.getNumBodies())),
                                          self.bodyIndex, j))

        super().__init__(bullet_client, base_link_name, list(range(bullet_client.getNumBodies())), self.bodyIndex, -1)

        # Init scene specific qualities
        self.body_name = object_name.decode("utf8")
        # The difference between these 2 is that a object might be removable,
        # but we might not want to reload it.
        # An example is a slice of an object. The original object is removable because we are removing it and
        # replacing it will slices of it. But we want to reload the original object, and not reload the slices
        self.removable = removable
        self.reloadable = reloadable
        self.removed = False

    def reload(self):
        # Load the body if it is currently none (was removed)
        if self.filename is not None and self.removed and self.removable and self.reloadable:
            print(f'Reloading {self.body_name}')
            self._p.loadURDF(**self.params)
            self.removed = False

    def remove(self):
        if self.removable and not self.removed:
            # So I think it doesnt like orphans
            self._p.removeBody(self.bodyIndex)
            self.removed = True

    def calc_state(self, scene):
        return 0


class SlicingSceneObject(SceneObject):
    def __init__(self, bullet_client: BulletClient, filename: str = None, position: list = None, orientation: list = None,
                 flags=0, removable=False, reloadable=True, slicing_parts: List[str] = None):
        super().__init__(bullet_client, filename, position, orientation, flags, removable, reloadable)

        # So we want to define: What parts can slice, and on what axis, direction, and deviation is that slicing valid
        if slicing_parts is None:
            self.slice_parts = [self] + self.children
        else:
            self.slice_parts = [_ for _ in [self] + self.children if _.body_name in slicing_parts]

    def calc_state(self, scene):
        pass


class SlicableSceneObject(SceneObject):
    def __init__(self, bullet_client: BulletClient, filename: str = None, position: list = None, orientation: list = None,
                 flags=0, removable=False, reloadable=True, create_body_params = None, create_collision_params = None,
                 create_visual_shape_params = None):
        """
        A slicable object for now is a single mesh, zero link object that can break into smaller objects
        via being removed from the world, and being replaced by smaller children.

        :param bullet_client:
        :param filename:
        :param position:
        :param orientation:
        :param flags:
        :param removable:
        :param
        reloadable:
        """
        self._p = bullet_client
        self.filename = filename
        if filename is None:
            create_visual_shape_params = {_: create_visual_shape_params[_] for _ in create_visual_shape_params if
                                          create_visual_shape_params[_] is not None}
            create_collision_params = {_: create_collision_params[_] for _ in create_collision_params if
                                          create_collision_params[_] is not None}
            create_body_params = {_: create_body_params[_] for _ in create_body_params if create_body_params[_] is not None}

            self.create_visual_shape_params = create_visual_shape_params
            self.create_collision_params = create_collision_params
            self.create_body_params = create_body_params

            self.removable = True
            self.reloadable = True
            self.removed = True
            self.reload()
            self.removable = removable
            self.reloadable = reloadable
            self.removed = False

        super().__init__(bullet_client, filename, position, orientation, flags, removable, reloadable)

    def reload(self):
        if self.filename is not None and self.removed and self.removable and self.reloadable:
            super(SlicableSceneObject, self).reload()
        elif self.removed and self.removable and self.reloadable:
            collision_id = self._p.createCollisionShape(**self.create_collision_params)
            visual_shape_id = self._p.createVisualShape(**self.create_visual_shape_params)
            self.create_body_params['baseCollisionShapeIndex'] = collision_id
            self.create_body_params['baseVisualShapeIndex'] = visual_shape_id
            self.bodyIndex = self._p.createMultiBody(**self.create_body_params)

    def calc_state(self, scene):
        """
        A slicable object will check if there are any Slicing objects that are contacting it.

        A few notes, a slicable object will check if a contacting object is able to slice.
        For example if a knife is cutting the object, but is backwards, then no cutting with actually happen.
        Therefore a Slicing object will have a set of axis that need to be aligned before it can slice.

        :param scene:
        :return:
        """
        if self.removed:
            return 0

        for slicing_object in [_ for _ in scene.scene_objects if type(_) is SlicingSceneObject
                               if not _.removed]:
            for slicing_joint in slicing_object.slice_parts:
                # Get a list of the overlapping objects
                collision = self._p.getOverlappingObjects(slicing_joint.get_position(), self.get_position())
                if collision is not None:
                    print(f'Collision is happening at {self.body_name}')
                    # Get the identifying ids for both interacting objects
                    slicing_joint_id = (slicing_joint.bodyIndex, slicing_joint.bodyPartIndex)
                    slicable_object_id = (self.bodyIndex, self.bodyPartIndex)
                    if slicing_joint_id in collision and slicable_object_id in collision:
                        print(f'Full Collision is happening at {self.body_name}')
                        """ Do Decomposition """
                        # Get information on slicable object
                        slicable_object_data = self._p.getCollisionShapeData(*slicable_object_id)[0]
                        slicable_object_visual_data = self._p.getVisualShapeData(slicable_object_id[0])
                        slicable_height, slicable_width, slicable_length = slicable_object_data[3]
                        slicable_position = self.get_position()
                        slicable_orientation = self.get_orientation()
                        slicable_base_mesh = slicable_object_data[3]
                        # Get information on slicing object
                        slicing_object_data = self._p.getCollisionShapeData(*slicing_joint_id)[0]
                        slicing_object_visual_data = self._p.getVisualShapeData(slicing_joint_id[0])
                        slicing_height, slicing_width, slicing_length = slicing_object_data[3]
                        slicing_position = slicing_joint.get_position()
                        slicing_orientation = slicing_joint.get_orientation()
                        slicing_base_mesh = slicing_object_data[3]

                        # Convert orientations into easier to read numeric types (degrees)
                        slicable_orientation_dg = np.rad2deg(self._p.getEulerFromQuaternion(slicable_orientation))
                        slicing_orientation_dg = np.rad2deg(self._p.getEulerFromQuaternion(slicing_orientation))
                        # Get the relative orientations. We cut relative to the slicable object
                        slicable_initial_orientation_dg = np.rad2deg(self._p.getEulerFromQuaternion(self.initialOrientation))
                        slicing_initial_orientation_dg = np.rad2deg(self._p.getEulerFromQuaternion(slicing_joint.initialOrientation))
                        slicable_rel_orientation_dg = slicable_initial_orientation_dg - slicable_orientation_dg
                        # Note, the reason this is this way, is because the knife might be init in a difference place
                        slicing_rel_orienation_dg = slicing_orientation_dg - slicing_initial_orientation_dg + slicable_rel_orientation_dg

                        if 80 < abs(slicing_rel_orienation_dg[0]) < 100 or 170 < abs(slicing_rel_orienation_dg[0]) or \
                             abs(slicing_rel_orienation_dg[0]) < 10:
                            print('Knife is rotated such that it can only realistically cut on the z axis')

                        # Create fake axis depending on rotation. Is there a better way to do this?
                        x_axis = 0 if 100 > abs(slicable_orientation_dg[1]) < 80 else 2
                        y_axis = 1 if 100 > abs(slicable_orientation_dg[0]) < 80 else 2
                        # Ok, so which axis does the cube need to be split?
                        if 80 < abs(slicing_rel_orienation_dg[2]) < 100 or 170 < abs(slicing_rel_orienation_dg[2]) or \
                             abs(slicing_rel_orienation_dg[2]) < 10:
                            # Set the split axis
                            sliceable_axis = y_axis if 80 < abs(slicing_rel_orienation_dg[0]) < 100 else x_axis
                            # Split the object
                            self.split(sliceable_axis, scene)

    def split(self, axis, scene):
        # Notes: the baseCollisionShapeIndex and baseVisualShapeIndex will automatically be added
        scene.scene_objects.append(SlicableSceneObject(self._p,
                                                       create_body_params={'baseMass':1,
                                                                           'basePosition': [0, 0, 0],
                                                                           'baseOrientation': [0, 0, 0, 1]},
                                                       create_collision_params={'shapeType': self._p.GEOM_BOX,
                                                                                'halfExtents': [0, 0, 0]},
                                                       create_visual_shape_params={'shapeType': self._p.GEOM_BOX,
                                                                                   'halfExtents': [0, 0, 0],
                                                                                   'rgbaColor': [1, 0, 0, 1],
                                                                                   'specularColor': [0.4, .4, 0]},
                                                       removable=True, reloadable=False))
        self.remove()