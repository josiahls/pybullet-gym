from collections import namedtuple
from typing import List

from namedlist import namedlist
from pybullet_utils.bullet_client import BulletClient
import numpy as np

# from .scene_manipulators import SceneFetch
from .robot_bases import BodyPart

Features = namedlist("Features", ['pos_x', 'pos_y', 'pos_z', 'or_x', 'or_y', 'or_z', 'width', 'height',
                                  'length', 'radius', 'obj_type', 'obj_internal_state'], default=0)


class SceneObject(BodyPart):
    def __init__(self, bullet_client: BulletClient, filename: str = None, position: list = None,
                 orientation: list = None,
                 flags=0, removable=False, reloadable=True, type_id=0):
        self.filename = filename
        if self.filename is not None:
            # Load the object mesh
            self.params = {'fileName': filename, 'basePosition': position, 'baseOrientation': orientation,
                           'flags': flags}
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
        self.type_id = type_id
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

    def calc_state(self, scene) -> namedtuple:
        """
        Updates the external and internal state of the scene object.

        :param scene:
        :return:
        """
        features = Features()

        if self.removed:
            return tuple(features)

        features.pos_x = self.get_position()[0]
        features.pos_y = self.get_position()[1]
        features.pos_z = self.get_position()[2]

        features.or_x = self._p.getEulerFromQuaternion(self.get_orientation())[0]
        features.or_y = self._p.getEulerFromQuaternion(self.get_orientation())[1]
        features.or_z = self._p.getEulerFromQuaternion(self.get_orientation())[2]

        # Get the collision information (description of the basic mesh)
        collision_info = self._p.getCollisionShapeData(self.bodyIndex, self.bodyPartIndex)
        if not collision_info:
            collision_info = [[0, 0, 0, (0, 0, 0)]]

        features.width = collision_info[0][3][0]
        features.height = collision_info[0][3][1]
        features.length = collision_info[0][3][2]
        features.radius = collision_info[0][3][0]

        # Set the types
        features.obj_type = self.type_id

        # Set the internal state
        features.obj_internal_state = self._get_internal_state(scene)

        # Before returning, we use the namedtuple to verify that all the field required are filled
        return tuple(features)

    def _get_internal_state(self, scene) -> float:
        """
        Each object might have some internal state it might want to modify such as distance, stress, decay, etc.

        :param scene:
        :return:
        """
        return 1.0

    def _closest_index_array_within_limit(self, array, value, limit):
        """
        This is a utility function for getting the closest index to a value within a limit.

        The goal of this method to for making finding axis closest to 0 and 90 degrees a little easier.

        Ignores negative / positive values. Treats calculations in absolute value space.

        :param array:
        :param value:
        :param limit:
        :return: -1 if no axis is close enough to the value within the given limit, else the index
        """
        index = (np.abs(array - value)).argmin()
        return index if abs(array[index]) < limit else -1

    def _is_value_within_limit(self, array, index, value, limit):
        """

        :param eval:
        :param index
        :param value:
        :param limit:
        :return:
        """
        return index if abs(np.abs(array[index]) - value) < limit else -1


class TargetSceneObject(SceneObject):
    def __init__(self, bullet_client: BulletClient, filename: str = None, position: list = None,
                 orientation: list = None,
                 flags=0, removable=False, reloadable=True, create_body_params=None, create_collision_params=None,
                 create_visual_shape_params=None, randomize=False, random_range=None):
        """

        Args:
            bullet_client:
            filename:
            position:
            orientation:
            flags:
            removable:
            reloadable:
            create_body_params:
            create_collision_params:
            create_visual_shape_params:
            randomize: True or False if we want to randomize the target position
            random_range: Should be a tuple in the format Tuple((x_max, y_max, z_max), (x_min, y_min, z_min)) or as
                          a single Tuple(x, y, z) that does randomized half extends from the origin
        """
        self.random_range = random_range
        self.randomize = randomize
        self._p = bullet_client
        self.filename = filename
        self.objects_to_compare = []
        if filename is None:
            create_visual_shape_params = {_: create_visual_shape_params[_] for _ in create_visual_shape_params if
                                          create_visual_shape_params[_] is not None}
            create_collision_params = {_: create_collision_params[_] for _ in create_collision_params if
                                       create_collision_params[_] is not None}
            create_body_params = {_: create_body_params[_] for _ in create_body_params if
                                  create_body_params[_] is not None}

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

        super().__init__(bullet_client, filename, position, orientation, flags, removable, reloadable, 1)

    def reload(self):
        super(TargetSceneObject, self).reload()
        # Target Objects have the option to randomize their positions
        if self.randomize and self.random_range is not None:
            pos = self.params['basePosition']
            self.reset_position((
                np.random.uniform(pos[0] - self.random_range[0], pos[0] + self.random_range[0]),
                np.random.uniform(pos[1] - self.random_range[1], pos[1] + self.random_range[1]),
                np.random.uniform(pos[2] - self.random_range[2], pos[2] + self.random_range[2])
            ))

    def set_objects_to_compare(self, objects_to_compare):
        self.objects_to_compare = objects_to_compare

    def _get_internal_state(self, scene):
        """
        Returns the sum distances of the target against non-target objects

        :param scene:
        :return:
        """
        objects_to_compare = [o for o in scene.scene_objects if type(o) is not TargetSceneObject]

        return sum([np.linalg.norm(self.get_position() - other_object.get_position())
                    for other_object in objects_to_compare])


class ProjectileSceneObject(SceneObject):
    def __init__(self, bullet_client: BulletClient, filename: str = None, position: list = None,
                 orientation: list = None,
                 flags=0, removable=False, reloadable=True, create_body_params=None, create_collision_params=None,
                 create_visual_shape_params=None, randomize=False, random_range=None):
        """

        Args:
            bullet_client:
            filename:
            position:
            orientation:
            flags:
            removable:
            reloadable:
            create_body_params:
            create_collision_params:
            create_visual_shape_params:
            randomize: True or False if we want to randomize the target position
            random_range: Should be a tuple in the format Tuple((x_max, y_max, z_max), (x_min, y_min, z_min)) or as
                          a single Tuple(x, y, z) that does randomized half extends from the origin
        """
        self.random_range = random_range
        self.randomize = randomize
        self._p = bullet_client
        self.filename = filename
        self.objects_to_compare = []
        if filename is None:
            create_visual_shape_params = {_: create_visual_shape_params[_] for _ in create_visual_shape_params if
                                          create_visual_shape_params[_] is not None}
            create_collision_params = {_: create_collision_params[_] for _ in create_collision_params if
                                       create_collision_params[_] is not None}
            create_body_params = {_: create_body_params[_] for _ in create_body_params if
                                  create_body_params[_] is not None}

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

        super().__init__(bullet_client, filename, position, orientation, flags, removable, reloadable, 1)

    def reload(self):
        super(ProjectileSceneObject, self).reload()
        # Target Objects have the option to randomize their positions
        if self.randomize and self.random_range is not None:
            pos = self.params['basePosition']
            self.reset_position((
                np.random.uniform(pos[0] - self.random_range[0], pos[0] + self.random_range[0]),
                np.random.uniform(pos[1] - self.random_range[1], pos[1] + self.random_range[1]),
                np.random.uniform(pos[2] - self.random_range[2], pos[2] + self.random_range[2])
            ))

    def set_objects_to_compare(self, objects_to_compare):
        self.objects_to_compare = objects_to_compare

    def _get_internal_state(self, scene):
        """
        Returns the sum distances of the target against non-target objects

        :param scene:
        :return:
        """
        objects_to_compare = [o for o in scene.scene_objects if type(o) is not TargetSceneObject]

        return sum([np.linalg.norm(self.get_position() - other_object.get_position())
                    for other_object in objects_to_compare])


class SlicingSceneObject(SceneObject):
    def __init__(self, bullet_client: BulletClient, filename: str = None, position: list = None,
                 orientation: list = None,
                 flags=0, removable=False, reloadable=True, slicing_parts: List[str] = None):
        super().__init__(bullet_client, filename, position, orientation, flags, removable, reloadable, 2)

        # So we want to define: What parts can slice, and on what axis, direction, and deviation is that slicing valid
        if slicing_parts is None:
            self.slice_parts = [self] + self.children
        else:
            self.slice_parts = [_ for _ in [self] + self.children if _.body_name in slicing_parts]


class SlicableSceneObject(SceneObject):
    def __init__(self, bullet_client: BulletClient, filename: str = None, position: list = None,
                 orientation: list = None,
                 flags=0, removable=False, reloadable=True, create_body_params=None, create_collision_params=None,
                 create_visual_shape_params=None):
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
        self.slicing_cool_down = 5
        if filename is None:
            create_visual_shape_params = {_: create_visual_shape_params[_] for _ in create_visual_shape_params if
                                          create_visual_shape_params[_] is not None}
            create_collision_params = {_: create_collision_params[_] for _ in create_collision_params if
                                       create_collision_params[_] is not None}
            create_body_params = {_: create_body_params[_] for _ in create_body_params if
                                  create_body_params[_] is not None}

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

        super().__init__(bullet_client, filename, position, orientation, flags, removable, reloadable, 3)

    def reload(self):
        if self.filename is not None and self.removed and self.removable and self.reloadable:
            self.slicing_cool_down = 5
            super(SlicableSceneObject, self).reload()
        elif self.removed and self.removable and self.reloadable:
            collision_id = self._p.createCollisionShape(**self.create_collision_params)
            visual_shape_id = self._p.createVisualShape(**self.create_visual_shape_params)
            self.create_body_params['baseCollisionShapeIndex'] = collision_id
            self.create_body_params['baseVisualShapeIndex'] = visual_shape_id
            self.bodyIndex = self._p.createMultiBody(**self.create_body_params)
            self.slicing_cool_down = 5

    def calc_state(self, scene):
        """
        A slicable object will check if there are any Slicing objects that are contacting it.

        A few notes, a slicable object will check if a contacting object is able to slice.
        For example if a knife is cutting the object, but is backwards, then no cutting with actually happen.
        Therefore a Slicing object will have a set of axis that need to be aligned before it can slice.

        :param scene:
        :return:
        """
        if not self.removed and self.slicing_cool_down != 0:
            self.slicing_cool_down -= 1
        elif not self.removed:
            # If the object has cooled down, start checking for slicing events
            for slicing_object in [_ for _ in scene.scene_objects if type(_) is SlicingSceneObject
                                   if not _.removed]:
                for slicing_joint in slicing_object.slice_parts:
                    # Get a list of the overlapping objects
                    collision = self._p.getOverlappingObjects(slicing_joint.get_position(), self.get_position())
                    if collision is not None:
                        # print(f'Collision is happening at {self.body_name}')
                        # Get the identifying ids for both interacting objects
                        slicing_joint_id = (slicing_joint.bodyIndex, slicing_joint.bodyPartIndex)
                        slicable_object_id = (self.bodyIndex, self.bodyPartIndex)
                        # Get the contact points
                        contacts_to_sliceable = self._p.getContactPoints(slicable_object_id[0], slicing_joint_id[0],
                                                                         slicable_object_id[1], slicing_joint_id[1])
                        contacts_to_slicing = self._p.getContactPoints(slicing_joint_id[0], slicable_object_id[0],
                                                                       slicing_joint_id[1], slicable_object_id[1])

                        # contacts_to_sliceable[0][11] < -0.8 is the specific downward force that the knife (body B)
                        # is applying to a slicable object A. This prevents "side" slicing. Only the sharp part
                        # of the blade is valid
                        if len(contacts_to_sliceable) == 1 and contacts_to_slicing[0][9] > 0.4 and abs(
                                contacts_to_slicing[0][11][1]) > 0.8:
                            # print(f'Full Collision is happening at {self.body_name}')
                            """ Do Decomposition """
                            # Get information on slicable object
                            slicable_object_data = self._p.getCollisionShapeData(*slicable_object_id)[0]
                            slicable_position = self.get_position()
                            slicable_orientation = self._p.getEulerFromQuaternion(self.get_orientation())
                            slicable_base_mesh = slicable_object_data[3]
                            # Get slicing orientation
                            slicing_orientation = self._p.getEulerFromQuaternion(slicing_joint.get_orientation())
                            # Get slicing relative orientation
                            slicing_rel_orientation = np.subtract(slicing_orientation, slicable_orientation)
                            axis = 0
                            """ Singular axis rotations (others are 0) """
                            # Y rotated 90 degrees, keep Y whether it is 90 or 0
                            axis = 0 if self._is_value_within_limit(slicing_rel_orientation, 0, 1.5,
                                                                    0.2) != -1 else axis
                            # X rotated 90 degrees, make Z
                            axis = 2 if self._is_value_within_limit(slicing_rel_orientation, 1, 1.5,
                                                                    0.2) != -1 else axis
                            # Z rotated 90 degrees, make X
                            axis = 1 if self._is_value_within_limit(slicing_rel_orientation, 2, 1.5,
                                                                    0.2) != -1 else axis
                            """ Combined axis rotations """
                            # For Y
                            axis = 2 if self._is_value_within_limit(slicing_rel_orientation, 0, 1.5, 0.2) != -1 and \
                                        self._is_value_within_limit(slicing_rel_orientation, 1, 1.5,
                                                                    0.2) != -1 else axis
                            axis = 1 if self._is_value_within_limit(slicing_rel_orientation, 0, 1.5, 0.2) != -1 and \
                                        self._is_value_within_limit(slicing_rel_orientation, 2, 1.5,
                                                                    0.2) != -1 else axis
                            # For X
                            axis = 1 if self._is_value_within_limit(slicing_rel_orientation, 1, 1.5, 0.2) != -1 and \
                                        self._is_value_within_limit(slicing_rel_orientation, 2, 1.5,
                                                                    0.2) != -1 else axis
                            axis = 2 if self._is_value_within_limit(slicing_rel_orientation, 1, 1.5, 0.2) != -1 and \
                                        self._is_value_within_limit(slicing_rel_orientation, 0, 1.5,
                                                                    0.2) != -1 else axis
                            axis = 2 if self._is_value_within_limit(slicing_rel_orientation, 0, 1.5, 0.2) != -1 and \
                                        self._is_value_within_limit(slicing_rel_orientation, 1, 1.5, 0.2) != -1 and \
                                        self._is_value_within_limit(slicing_rel_orientation, 2, 1.5,
                                                                    0.2) != -1 else axis

                            # Finally, if the knive's contact point's normal is parallel to the axis, then
                            if axis == np.argmax(np.abs(contacts_to_slicing[0][7])):
                                # print('The knife is not allowed to cut perpendicular to its blade direction')
                                break

                            self._split(axis, slicable_base_mesh, slicable_position, slicable_orientation, scene,
                                        contacts_to_sliceable)

        return super(SlicableSceneObject, self).calc_state(scene)

    def _split(self, axis, original_mesh, original_position, original_orientation, scene, contacts):
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
        slice2_position[axis] += (1.1 * slice_mesh2[axis])
        # Notes: the baseCollisionShapeIndex and baseVisualShapeIndex will automatically be added
        self.remove()
        scene.scene_objects.append(SlicableSceneObject(self._p,
                                                       create_body_params={'baseMass': 1,
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
                                                       create_body_params={'baseMass': 1,
                                                                           'basePosition': slice2_position,
                                                                           'baseOrientation': original_orientation},
                                                       create_collision_params={'shapeType': self._p.GEOM_BOX,
                                                                                'halfExtents': slice_mesh2},
                                                       create_visual_shape_params={'shapeType': self._p.GEOM_BOX,
                                                                                   'halfExtents': slice_mesh2,
                                                                                   'rgbaColor': [1, 0, 0, 1],
                                                                                   'specularColor': [0.4, .4, 0]},
                                                       removable=True, reloadable=False))
