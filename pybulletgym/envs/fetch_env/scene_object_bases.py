import pybullet

from pybullet_envs.scene_abstract import Scene
from pybullet_utils.bullet_client import BulletClient
from robot_bases import BodyPart
import os


class SceneObject(BodyPart):

    def __init__(self, bullet_client: BulletClient, filename: str, position: list = None, orientation: list = None,
                 flags=0, keep_on_reset=True):
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
        self.keep_on_reset = keep_on_reset

    def reload(self):
        # Load the body if it is currently none (was removed)
        try:
            if self.keep_on_reset and (self.bodyIndex >= self._p.getNumBodies() or
                                       self._p.getBodyInfo(self.bodyIndex) is None):
                self._p.loadURDF(**self.params)
        except pybullet.error:
            self._p.loadURDF(**self.params)

    def calc_state(self, scene: Scene):
        pass
