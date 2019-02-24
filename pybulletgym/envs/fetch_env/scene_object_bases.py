from pybullet_utils.bullet_client import BulletClient
from robot_bases import BodyPart


class SceneObject(BodyPart):

    def __init__(self, bullet_client: BulletClient, body_index, keep_on_reset=True):
        # Init the super object
        base_link_name, object_name = bullet_client.getBodyInfo(body_index)
        base_link_name = base_link_name.decode("utf8")
        # TODO Do something about the part index?
        super().__init__(bullet_client, base_link_name, list(range(self._p.getNumBodies())), body_index, -1)

        # Init scene specific qualities
        self.object_name = object_name.decode("utf8")
        self.keep_on_reset = keep_on_reset

    def calc_state(self):
        pass
