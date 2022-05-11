from .data_manager import DataManager
from .instance_object import InstanceObject
from .image_object import ImageObject
import json, os, copy
import numpy as np

def load_from_json(json_file, mode='pred'):
    """ Json format from MI output """
    assert mode in ['pred', 'gth']
    assert os.path.exists(json_file)
    with open(json_file, 'r') as fid:
        json_obj = json.load(fid)

    # deal with class dict
    label2class = {}
    for class_obj in json_obj['class_dict']:
        class_id = class_obj['class_id']-1
        label2class[class_id] = class_obj['class_name']

    class_ids = sorted(label2class.keys())
    assert all([x==y for x, y in zip(class_ids, range(len(class_ids)))])  # check labels enumerate from 0, 1, ..., #classes-1

    image_object_dict = dict()
    for item in json_obj['record']:
        # grab distribution
        distribution = np.zeros(len(label2class))
        if mode == 'pred':
            assert len(item['pred_inst']) == len(label2class), "json error: prediction class not fit with the class-dict: {}".format(json_file)
            for cls_obj in item['pred_inst']:
                class_id = cls_obj['class_id'] - 1
                assert class_id in class_ids, "Invalid json, class id unmatches with class_dict"
                distribution[class_id] = cls_obj['score']
        else:
            assert len(item['gt_inst']) > 0
            for gt_class in item['gt_inst']:  # in multi-label data, gt_inst could have multiple classes
                class_id = gt_class['class_id'] - 1
                assert class_id in class_ids, "Invalid json, class id {} unmatches with class_dict: {}".format(class_id, class_ids)
                distribution[class_id] = 1.

        # assert np.abs(distribution.sum()- 1.) < 1e-6, "the distribution should sum up to 1"  # this not apply to multilabel

        instance_object = InstanceObject(
            box = tuple(),  # empty type for image level annotation
            distribution=distribution
        )

        image_name = item['info']['image_path']
        image_object_dict[image_name] = ImageObject(
            image_name=image_name,
            instance_object_list=[instance_object],
        )

    return image_object_dict, label2class

class ClassificationData(DataManager):
    """
    Data class for classification results:
    ClassificationData.label2class: manifest all classes, whose labels START FROM ZERO (In classification, there is no conceptual difference between fg and bg)
    Note: this is different from DetectionData.label2class, where labels start from ONE, because there is no position for the bg class in label2class
    """
    def compatible_with(self, obj):        
        """
        check if the input model preds target on the same data, and so well on the same classes,
        if this verification fails, return False
        """
        assert isinstance(obj, self.__class__)
        # check class dict
        if self.label2class != obj.label2class:
            return False
        # check class names
        image_names_pred1 = set(self.image_names)
        image_names_pred2 = set(obj.image_names)
        return image_names_pred1 == image_names_pred2

    @classmethod
    def from_pred_json(cls, json_file):
        """ Since the given json contained both gth and classification preds, this method loads the pred part only """
        image_object_dict, label2class = load_from_json(json_file, mode='pred')
        return cls(image_object_dict, label2class)

    @classmethod
    def from_gth_json(cls, json_file):
        """ Since the given json contained both gth and classification preds, this method loads the gth part only """
        image_object_dict, label2class = load_from_json(json_file, mode='gth')
        return cls(image_object_dict, label2class)

    @property
    def class_num(self):
        return len(self.label2class)

    def dump_info(self):
        """ print information of dataset """
        num_instances_per_class = dict()
        for image_object in self.image_object_dict.values():
            for instance_object in image_object.instance_object_list:
                class_name = self.label2class[np.argmax(instance_object.distribution)]
                if class_name in num_instances_per_class:
                    num_instances_per_class[class_name] += 1
                else:
                    num_instances_per_class[class_name] = 1
        print("#Images: {}".format(len(self.image_object_dict)))
        print(num_instances_per_class)


class MultiClassData(ClassificationData):
    pass


class MultiLabelData(ClassificationData):
    @classmethod
    def from_DetectionData(cls, detections, lowest_score=0):
        """ devolve annotations from instance-level data to image-level data """
        from .detection_manager import DetectionData
        assert isinstance(detections, DetectionData)
        # 1) mirror class dict
        label2class = copy.copy(detections.label2class)
        assert 0 not in label2class, "Ok class is not allowed in DetectionData class_dict: {}".format(label2class)
        assert min(label2class) == 1, label2class
        label2class[0] = 'ok'
        # 2) create image_object_dict
        image_object_dict = dict()
        for image_name in detections.image_names:
            img_obj_det = detections[image_name]
            if img_obj_det.instance_num > 0:
                multi_class_dists = img_obj_det.dists  # Assumming detection distribution is the multi-class type
                bg_dists = multi_class_dists[:,:1]
                fg_dists = multi_class_dists[:,1:]
                dist_multilabel = np.hstack([bg_dists.min(axis=0), fg_dists.max(axis=0)])  # use the maximal fg probs, minimal bg prob of all instances for image-level prob
            else:
                dist_multilabel = np.zeros(len(label2class)) + lowest_score
            img_obj = ImageObject(
                image_name = image_name,
                instance_object_list = [InstanceObject(box=tuple(), distribution=dist_multilabel)]
            )
            image_object_dict[image_name] = img_obj
        return cls(image_object_dict, label2class)