from .data_manager import DataManager
from .instance_object import InstanceObject
from .image_object import ImageObject
from .classification_manager import MultiLabelData
import json, os, copy
import numpy as np

class DetectionData(DataManager):
    """ Data class for Detection Results """
    def compatible_with(self, obj):        
        """
        check if the input model preds target on the same data, and so well on the same classes,
        if this verification fails, return False
        """
        assert isinstance(obj, self.__class__) or isinstance(obj, MultiLabelData)

        if isinstance(obj, MultiLabelData):  # In hybrid ensemble case: Detection + MultiLabel = Detection
            label2class = copy.copy(obj.label2class)
            if 0 in label2class:
                label2class.pop(0)  # pop the background (in this case) class from the dict
            else:
                assert label2class == {}, "Multilabel class dict should contain class_id 0"
        else:
            label2class = obj.label2class

        # check class dict
        if self.label2class != label2class:
            return False
        # check class names
        image_names_pred1 = set(self.image_names)
        image_names_pred2 = set(obj.image_names)
        return image_names_pred1.issubset(image_names_pred2) and image_names_pred2.issubset(image_names_pred1)

    @property
    def class_num(self):
        return len(self.label2class)+1  # #bg-classes + 1 (fg)

    @classmethod
    def from_json(cls, json_file):
        """ Iniatilize class from json file """
        assert os.path.exists(json_file)
        with open(json_file, 'r') as fid:
            json_obj = json.load(fid)

        # deal with class dict
        label2class = {}
        for class_obj in json_obj['class_dict']:
            label2class[class_obj['class_id']] = class_obj['class_name']

        image_dict = dict()
        for item in json_obj['image']:
            image_dict[item['image_id']] = item

        instance_dict = dict()
        for item in json_obj['instance']:
            image_id = item['image_id']
            image_name = image_dict[image_id]['image_name']
            x, y, w, h = item['instance_bbox']
            assert w > 0 and h > 0, "Invalid bbox {} found in {}".format(item['instance_bbox'], image_name)

            # load distribution
            if 'instance_scores' in item:  # in prediction json
                distribution = np.array(item['instance_scores'])
                assert len(distribution) == len(label2class)+1, "class dict and instance distribution unmatch. Dist: {}".format(distribution)
            else:  # gth doesnot have distribution label
                distribution = np.zeros(len(label2class)+1)
                assert "class_id" in item
                distribution[item["class_id"]] = 1.

            instance_object = InstanceObject(
                box=(x, y, x+w, y+h),
                distribution=distribution
                )
            if image_id in instance_dict:
                instance_dict[image_id].append(instance_object)
            else:
                instance_dict[image_id] = [instance_object]

        # generate list of ImageObject
        image_object_dict = dict()
        for image_id, image_info in image_dict.items():
            if image_id in instance_dict:
                instance_object_list = instance_dict[image_id]
            else:
                instance_object_list = []

            assert image_info['image_name'] not in image_object_dict
            image_object_dict[image_info['image_name']] = \
                ImageObject(
                    image_name=image_info['image_name'],
                    instance_object_list=instance_object_list,
                )

        return cls(image_object_dict, label2class)

    def save(self, save_path):
        """
        Save current class instance to json_file in the format of annotation jsons
        param:
            save_path: json path you want to save to
        """
        # turn self.image_object_dict to image_info and instance_info
        image_info, instance_info = list(), list()
        for image_object in self.image_object_dict.values():
            img_json_obj = image_object.json_format()
            img_json_obj.update({'image_id': len(image_info)+1})
            image_info.append(img_json_obj)
            # add instance objects on the image to instance_info
            for instance_object in image_object.instance_object_list:
                inst_json_obj = instance_object.json_format()
                inst_json_obj['image_id'] = img_json_obj['image_id']
                instance_info.append(inst_json_obj)

        # create json object
        json_obj = dict()
        json_obj['image'] = image_info
        json_obj['instance'] = instance_info
        json_obj['class_dict'] = list()
        for class_id, class_name in self.label2class.items():
            json_obj['class_dict'].append(
                {'class_id': class_id, 'class_name': class_name}
            )

        # save json to path
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_path, 'w') as fid:
            json.dump(json_obj, fid, indent=4)

    def dump_info(self, file=None):
        """ print information of dataset """
        num_ok_images = 0
        num_ng_images = 0
        num_instances = 0
        num_instances_per_class = dict()
        for image_object in self.image_object_dict.values():
            if image_object.instance_num > 0:
                num_ng_images += 1
            else:
                num_ok_images += 1
            num_instances += image_object.instance_num

            for instance_object in image_object.instance_object_list:
                class_name = self.label2class[np.argmax(instance_object.distribution[1:])+1]
                if class_name in num_instances_per_class:
                    num_instances_per_class[class_name] += 1
                else:
                    num_instances_per_class[class_name] = 1
        basic_info = "#Ok Images: {}    #Ng Images: {}    #Instances: {}".format(num_ok_images, num_ng_images, num_instances)
        if file is None:
            print(basic_info)
            print(num_instances_per_class)
        else:
            print(basic_info, file=open(file,'w'))
            print(num_instances_per_class, file=open(file,'a'))