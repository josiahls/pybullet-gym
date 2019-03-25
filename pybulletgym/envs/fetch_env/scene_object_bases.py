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
            # We want to make sure that the slot for this body is empty
            self._p.removeBody(self.bodyIndex)
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
                    # Get the contact points
                    contacts_to_sliceable = self._p.getContactPoints(slicable_object_id[0], slicing_joint_id[0],
                                                                     slicable_object_id[1], slicing_joint_id[1])
                    contacts_to_slicing = self._p.getContactPoints(slicing_joint_id[0], slicable_object_id[0],
                                                                   slicing_joint_id[1], slicable_object_id[1])

                    # contacts_to_sliceable[0][11] < -0.8 is the specific downward force that the knife (body B) is applying to
                    # a slicable object A. This prevents "side" slicing. Only the sharp part of the blade is valid
                    if len(contacts_to_sliceable) > 0 and contacts_to_slicing[0][7][2] > 0.3:
                        print(f'Full Collision is happening at {self.body_name}')
                        """ Do Decomposition """
                        # Get information on slicable object
                        slicable_object_data = self._p.getCollisionShapeData(*slicable_object_id)[0]
                        slicable_position = self.get_position()
                        slicable_orientation = self.get_orientation()
                        slicable_base_mesh = slicable_object_data[3]

                        self.split(np.argmax(np.abs(contacts_to_sliceable[0][11])),
                                   slicable_base_mesh, slicable_position,
                                   slicable_orientation, scene, contacts_to_sliceable)

    def split(self, axis, original_mesh, original_position, original_orientation, scene, contacts):
        print(str(scene.scene_objects))

        original_mesh = list(map(lambda x: x * 0.5, original_mesh))
        # The distributions is the percent overlap the contact has on the cube width
        distribution = abs((contacts[0][5][axis] - original_position[axis]) / original_mesh[axis])
        distribution_dir = np.sign((contacts[0][5][axis] - original_position[axis]))

        # There are differences in x, y directions. We want to handle these differently
        distribution = distribution if axis == 1 else (1 - distribution)
        distribution = distribution if distribution_dir < 0 else (1 - distribution)

        slice_mesh1 = np.copy(original_mesh)
        slice_mesh2 = np.copy(original_mesh)
        # We take the width * half (because half extends) * the percent
        slice_mesh1[axis] = slice_mesh1[axis] * distribution
        slice_mesh2[axis] = slice_mesh2[axis] * (1 - distribution)

        slice1_position = np.copy(original_position)
        slice1_position[axis] = contacts[0][5][axis]
        slice1_position[axis] += (-1.1 * slice_mesh1[axis])

        slice2_position = np.copy(original_position)
        slice2_position[axis] = contacts[0][5][axis]
        slice2_position[axis] += (1.1*slice_mesh2[axis])
        # Notes: the baseCollisionShapeIndex and baseVisualShapeIndex will automatically be added
        self.remove()
        scene.scene_objects.append(SlicableSceneObject(self._p,
                                                       create_body_params={'baseMass':1,
                                                                           'basePosition': slice1_position,
                                                                           'baseOrientation': original_orientation},
                                                       create_collision_params={'shapeType': self._p.GEOM_BOX,
                                                                                'halfExtents': slice_mesh1},
                                                       create_visual_shape_params={'shapeType': self._p.GEOM_BOX,
                                                                                   'halfExtents': slice_mesh1,
                                                                                   'rgbaColor': [1, 0, 0, 1],
                                                                                   'specularColor': [0.4, .4, 0]},
                                                       removable=True, reloadable=False))

        scene.scene_objects.append(SlicableSceneObject(self._p,
                                                       create_body_params={'baseMass':1,
                                                                           'basePosition': slice2_position,
                                                                           'baseOrientation': original_orientation},
                                                       create_collision_params={'shapeType': self._p.GEOM_BOX,
                                                                                'halfExtents': slice_mesh2},
                                                       create_visual_shape_params={'shapeType': self._p.GEOM_BOX,
                                                                                   'halfExtents': slice_mesh2,
                                                                                   'rgbaColor': [1, 0, 0, 1],
                                                                                   'specularColor': [0.4, .4, 0]},
                                                       removable=True, reloadable=False))
