#
# A manager that maintains json for training
#
import os, json, copy
import numpy as np

class DataManager(object):
    def __init__(self, image_object_dict, label2class):
        """ Initialize DataManager """
        assert isinstance(image_object_dict, dict)
        assert isinstance(label2class, dict)
        self.image_object_dict = image_object_dict
        self.label2class = label2class
        self.class2label = {y:x for x,y in self.label2class.items()}

    def compatible_with(self, obj):        
        """
        check if the input model preds target on the same data, and so well on the same classes,
        if this verification fails, return False
        """
        raise NotImplementedError

    @classmethod
    def class_num(self):
        raise NotImplementedError

    @classmethod
    def from_json(cls, json_file):
        raise NotImplementedError

    def save(self, save_path):
        raise NotImplementedError

    @property
    def image_names(self):
        """ iterate all image names in image_object_dict """
        for image_name in self.image_object_dict:
            yield image_name

    def __getitem__(self, image_name):
        """ index image object by image name """
        if image_name not in self.image_object_dict:
            raise KeyError("'{}' not found in image object dict".format(image_name))
        return self.image_object_dict[image_name]

    def clone(self):
        """ clone current object """
        return copy.deepcopy(self)