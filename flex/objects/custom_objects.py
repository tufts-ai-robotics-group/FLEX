from robosuite.models.objects import MujocoXMLObject
from utils.sim_utils import fill_custom_xml_path
from utils.model_uilts import scale_object_new
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion
import numpy as np
import os


class DrawerObject(MujocoXMLObject):
    """
    Door with handle (used in Drawer)

    Args:
    """

    def __init__(self, name):
        xml_path = "drawer/drawer.xml"
        super().__init__(
            fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all", duplicate_collision_geoms=True
        )

        # Set relevant body names
        self.drawer_body = self.naming_prefix + "link_1"
        self.drawer_joint = self.naming_prefix + "joint_1"

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        return dic
    
class SelectedDrawerObject(MujocoXMLObject):
    def __init__(self, name, xml_path):
        super().__init__(
            fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all", duplicate_collision_geoms=True
        )

        # Set relevant body names
        self.drawer_body = self.naming_prefix + "link_1"
        self.drawer_joint = self.naming_prefix + "joint_1"


class OriginalDoorObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """
    def __init__(self, name, friction=None, damping=None, lock=False):
        xml_path = "door_original/door_original.xml"
        super().__init__(
            fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all", duplicate_collision_geoms=True
        )
        self.door_body = self.naming_prefix + "door"
        self.frame_body = self.naming_prefix + "frame"
        # self.latch_body = self.naming_prefix + "latch"
        self.hinge_joint = self.naming_prefix + "hinge"
        self.lock = False # we don't have lock for this door
        self.friction = friction
        self.damping = damping

        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)
    
    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML

        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML

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

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        dic.update({"hinge": self.naming_prefix + "hinge_site"})
        return dic
    
    @property
    def door_panel_size(self):
        '''
        Returns:
            tuple: size of the door panel
        '''

        # TODO: hard-coded value for now, is there a way to extract this from the XML? 
        return (0.22, 0.02, .29)
    
    @property
    def hinge_direction(self):
        '''
        Returns:
            str: hinge direction
        '''
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge_direction_text = hinge.get("axis")
        hinge_direction = hinge_direction_text.split(" ")
        hinge_direction = [float(x) for x in hinge_direction]
        return hinge_direction
    
    
    
class SelectedMicrowaveObject(MujocoXMLObject):
    """
    Microwave with door (used in Microwave)

    Args:
    """

    def __init__(self, name, microwave_number, scale=False):

        available_numbers = [1, 2, 3, 4, 5]
        assert microwave_number in available_numbers, "Microwave number must be one of {}".format(available_numbers)

        xml_path = f"microwave-{microwave_number}/microwave-{microwave_number}.xml" if not scale else f'microwave-{microwave_number}/rescaled-microwave-{microwave_number}.xml'
        super().__init__(
            fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all", duplicate_collision_geoms=True
        )

        # Set relevant body names
        self.revolute_body = self.naming_prefix + "link_0"
        self.hinge_joint = self.naming_prefix + "joint_0"
    @property
    def hinge_pos_relative(self):
        '''
        Returns:
            str: hinge position relative to the object
        '''
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge_pos = hinge.get("pos")
        hinge_pos = hinge_pos.split(" ")
        hinge_pos = [float(x) for x in hinge_pos]
        return hinge_pos
    

class SelectedDishwasherObject(MujocoXMLObject):
    """
    Microwave with door (used in Microwave)

    Args:
    """

    def __init__(self, name, scaled=False, scale=1.0,):


        self.available_obj_list = self.available_objects()
        assert name in self.available_obj_list, "Invalid object! Please use another name"
        if not scaled:
            xml_path = f"{name}/{name}.xml"
        
        else:
            # call function to scale the object
            xml_path_original = fill_custom_xml_path(f"{name}/{name}.xml")
            scale_object_new(xml_path_original, None, [scale, scale, scale])
            xml_path = f"{name}/{name}-scaled.xml"
        
        super().__init__(
            fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all",duplicate_collision_geoms=True
            )
        
        # Set relevant body names
        self.revolute_body = self.naming_prefix + "link_0"
        self.hinge_joint = self.naming_prefix + "joint_0"
    
    @property
    def hinge_pos_relative(self):
        '''
        Returns:
            str: hinge position relative to the object
        '''
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge_pos = hinge.get("pos")
        hinge_pos = hinge_pos.split(" ")
        hinge_pos = [float(x) for x in hinge_pos]
        return hinge_pos

    @staticmethod
    def available_objects():
        available_objects = [
            'dishwasher-2',
            'dishwasher-3',
        ]
        return available_objects

class EvalRevoluteObjects(MujocoXMLObject):
    def __init__(self, name, scaled = True, scale = 1.0):
        # available_names = ["train-door-counterclock-1", "door_original", "train-microwave-1", "train-dishwasher-1"]
        assert name in self.available_objects(), "Object name must be one of {}".format(self.available_objects())
        
        if not scaled:
            xml_path = f"{name}/{name}.xml"
        
        else:
            # call function to scale the object
            xml_path_original = fill_custom_xml_path(f"{name}/{name}.xml")       
            scale_object_new(xml_path_original, None, [scale, scale, scale])
            xml_path = f"{name}/{name}-scaled.xml"
        super().__init__(
            fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all",duplicate_collision_geoms=True
            )

        # Set relevant body names
        if not name == "door_original":
            self.revolute_body = self.naming_prefix + "link_0"
            self.hinge_joint = self.naming_prefix + "joint_0"
        else:
            self.revolute_body = self.naming_prefix + "door"
            # self.frame_body = self.naming_prefix + "frame"
            self.hinge_joint = self.naming_prefix + "hinge"
            self.lock = False
            # set the door friction and damping to 0 for training
        self._set_door_friction(0.0)
        self._set_door_damping(0.0) 
    
    @staticmethod
    def available_objects():
        available_objects = [
            'microwave-1', 
            'microwave-2',
            'microwave-3',
            'microwave-4',
            'microwave-5',
            'microwave-6',
            # 'dishwasher-1',
            'dishwasher-2',
            'dishwasher-3', 
            'train-microwave-1', 
            'train-dishwasher-1',
        ]
        return available_objects

    def _set_door_friction(self, friction):
      
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):

        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))
    
    @property
    def hinge_direction(self):
        '''
        Returns:
            str: hinge direction
        '''
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge_direction_text = hinge.get("axis")
        hinge_direction = hinge_direction_text.split(" ")
        hinge_direction = [float(x) for x in hinge_direction]
        return hinge_direction 
    
    @property
    def hinge_pos_relative(self):
        '''
        Returns:
            str: hinge position relative to the object
        '''
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge_pos = hinge.get("pos")
        hinge_pos = hinge_pos.split(" ")
        hinge_pos = [float(x) for x in hinge_pos]
        return hinge_pos 
    
    @property
    def hinge_range(self):
        '''
        Returns:
            str: hinge ransge
        '''
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge_range = hinge.get("range")
        hinge_range = hinge_range.split(" ")
        hinge_range = [float(x) for x in hinge_range]
        return hinge_range



class TrainRevoluteObjects(MujocoXMLObject):
    """
    Sample Objects with revolute joints (used in Train)
    Args:
        name (str): Name of the object
    """

    def __init__(self, name, scaled=True, scale=1.0):
        # available_names = ["train-door-counterclock-1", "door_original", "train-microwave-1", "train-dishwasher-1"]
        assert name in self.available_objects(), "Object name must be one of {}".format(self.available_objects)

        # xml_path = f"{name}/{name}.xml"
        # super().__init__(
        #     fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all", duplicate_collision_geoms=True
        # )
        if not scaled:
            xml_path = f"{name}/{name}.xml"
        else:
            xml_path = f"{name}/{name}-scaled.xml"
        # else:
        #     # call function to scale the object
        #     xml_path_original = fill_custom_xml_path(f"{name}/{name}.xml")
        #     xml_path = f"{name}/{name}-scaled.xml"
        #     if not os.path.exists(xml_path):
        #        scale_object_new(xml_path_original, None, [scale, scale, scale])
            # xml_path = f"{name}/{name}-scaled.xml"
        super().__init__(
            fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all",duplicate_collision_geoms=True
            )

        # Set relevant body names
        if not name == "door_original":
            self.revolute_body = self.naming_prefix + "link_0"
            self.hinge_joint = self.naming_prefix + "joint_0"
        else:
            self.revolute_body = self.naming_prefix + "door"
            # self.frame_body = self.naming_prefix + "frame"
            self.hinge_joint = self.naming_prefix + "hinge"
            self.lock = False
            # set the door friction and damping to 0 for training
        self._set_door_friction(0.0)
        self._set_door_damping(0.0) 
    
    @staticmethod
    def available_objects():
        available_objects = [
            # 'train-drawer-1', 
            'train-dishwasher-1',
            'train-microwave-1',
            'dishwasher-2'
            # 'cabinet-1', 
            # 'cabinet-2', 
            # 'cabinet-3', 
        ]
        return available_objects

    def _set_door_friction(self, friction):
      
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):

        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))
    
    @property
    def hinge_direction(self):
        '''
        Returns:
            str: hinge direction
        '''
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge_direction_text = hinge.get("axis")
        hinge_direction = hinge_direction_text.split(" ")
        hinge_direction = [float(x) for x in hinge_direction]
        return hinge_direction
    
    @property
    def hinge_pos_relative(self):
        '''
        Returns:
            str: hinge position relative to the object
        '''
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge_pos = hinge.get("pos")
        hinge_pos = hinge_pos.split(" ")
        hinge_pos = [float(x) for x in hinge_pos]
        return hinge_pos
    
    @property
    def hinge_range(self):
        '''
        Returns:
            str: hinge ransge
        '''
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.hinge_joint}, return_first=True)
        hinge_range = hinge.get("range")
        hinge_range = hinge_range.split(" ")
        hinge_range = [float(x) for x in hinge_range]
        return hinge_range
    

class TrainPrismaticObjects(MujocoXMLObject):
    def __init__(self, name, scaled=True, scale=1.0):

        self.available_obj_list = self.available_objects()
        assert name in self.available_obj_list, "Invalid object! Please use another name"

        xml_path = f"{name}/{name}.xml"
        super().__init__(
            fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all",duplicate_collision_geoms=True
            )
        
        self.set_panel_size()
        self.prismatic_body = self.naming_prefix + "link_0"
        self.joint = self.naming_prefix + "joint_0"
        
    def set_panel_size(self):
        self.panel_geom_size = {
            "train-drawer-1": [(-0.,0.), (-0.1, 0.1), (-0.3, 0.3)],
        }

    @staticmethod
    def available_objects():
        available_objects = [
            'train-drawer-1', 
            # 'cabinet-1', 
            # 'cabinet-2', 
            # 'cabinet-3', 
        ]
        return available_objects
    
    @property
    def joint_pos_relative(self):
        joint = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.joint}, return_first=True)
        joint_pos = joint.get("pos")
        joint_pos = joint_pos.split(" ")
        joint_pos = [float(x) for x in joint_pos]
        return joint_pos
    
    @property
    def joint_range(self):
        '''
        Returns:
            str: hinge ransge
        '''
        joint = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.joint}, return_first=True)
        joint_range = joint.get("range")
        joint_range = joint_range.split(" ")
        joint_range = [float(x) for x in joint_range]
        return joint_range
    
    @property
    def joint_direction(self):
        '''
        Returns:
            str: joint direction
        '''
        joint = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.joint}, return_first=True)
        joint_direction = joint.get("axis")
        joint_direction = joint_direction.split(" ")
        joint_direction = [float(x) for x in joint_direction]
        return joint_direction
        
class EvalPrismaticObjects(MujocoXMLObject):
    def __init__(self, name, scaled = True, scale = 1.0):
        self.available_obj_list = self.available_objects()
        assert name in self.available_obj_list, "Invalid object! Please use another name"

        if not scaled:
            xml_path = f"{name}/{name}.xml"
        
        else:
            # call function to scale the object
            xml_path_original = fill_custom_xml_path(f"{name}/{name}.xml")
            scale_object_new(xml_path_original, None, [scale, scale, scale])
            xml_path = f"{name}/{name}-scaled.xml"
        super().__init__(
            fill_custom_xml_path(xml_path), name=name, joints=None, obj_type="all",duplicate_collision_geoms=True
            )
        self.prismatic_body = self.naming_prefix + "link_0"
        self.joint = self.naming_prefix + "joint_0"

    @staticmethod
    def available_objects():
        available_objects = [
            'trashcan-1',
            'trashcan-2', 
            'trashcan-3',
            'cabinet-1',
            'cabinet-2',
            'cabinet-3',
        ]
        return available_objects
    
    @property
    def joint_range(self):
        '''
        Returns:
            str: hinge ransge
        '''
        joint = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.joint}, return_first=True)
        joint_range = joint.get("range")
        joint_range = joint_range.split(" ")
        joint_range = [float(x) for x in joint_range]
        return joint_range 
    @property
    def joint_pos_relative(self):
        joint = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.joint}, return_first=True)
        joint_pos = joint.get("pos")
        joint_pos = joint_pos.split(" ")
        joint_pos = [float(x) for x in joint_pos]
        return joint_pos
    @property
    def joint_direction(self):
        '''
        Returns:
            str: joint direction
        '''
        joint = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.joint}, return_first=True)
        joint_direction = joint.get("axis")
        joint_direction = joint_direction.split(" ")
        joint_direction = [float(x) for x in joint_direction]
        return joint_direction