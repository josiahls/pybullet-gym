import pybullet

from pybullet_envs.scene_abstract import Scene
from pybullet_utils.bullet_client import BulletClient
from robot_bases import BodyPart
import os


class SceneObject(BodyPart):
    CURRENT_REMOVED_ORDER = -1

    def __init__(self, bullet_client: BulletClient, filename: str, position: list = None, orientation: list = None,
                 flags=0, removable=False, reloadable=True):
        # Load the object mesh
        self.params = {'fileName': filename, 'basePosition': position, 'baseOrientation': orientation, 'flags': flags}
        self.params = {_: self.params[_] for _ in self.params if self.params[_] is not None}
        body_index = bullet_client.loadURDF(**self.params)

        # Init the super object
        base_link_name, object_name = bullet_client.getBodyInfo(body_index)
        base_link_name = base_link_name.decode("utf8")
        # TODO Do something about the part index?
        super().__init__(bullet_client, base_link_name, list(range(bullet_client.getNumBodies())), body_index, -1)

        # Init scene specific qualities
        self.object_name = object_name.decode("utf8")
        # The difference between these 2 is that a object might be removable,
        # but we might not want to reload it.
        # An example is a slice of an object. The original object is removable because we are removing it and
        # replacing it will slices of it. But we want to reload the original object, and not reload the slices
        self.removable = removable
        self.reloadable = reloadable
        self.removed = False
        # We want to add the object is the same order that it was removed to avoid id conflicts
        self.removed_order = -1

    def reload(self):
        # Load the body if it is currently none (was removed)
        if self.removed and self.removable and self.reloadable:
            print(f'Reloading {self.object_name}')
            self._p.loadURDF(**self.params)
            self.removed = False

    def calc_state(self, scene: Scene):
        pass
