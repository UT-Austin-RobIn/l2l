import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, find_elements

import pathlib
absolute_path = pathlib.Path(__file__).parent.absolute()

class ButtonObject(MujocoXMLObject):
    def __init__(self, name, friction=None, damping=None):
        super().__init__(str(absolute_path) + "/" + "button.xml",
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

        # Set relevant body names
        self.hinge_joint = self.naming_prefix + "hinge"

        self.friction = friction
        self.damping = damping

    def _set_friction(self, friction):
        """
        Helper function to override the drawer friction directly in the XML

        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_damping(self, damping):
        """
        Helper function to override the drawer friction directly in the XML

        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of drawer handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({
            "handle": self.naming_prefix + "handle"
        })
        return dic
