import numpy as np
from tqdm import tqdm
import os, collections
from me_utils import voc_ap, compute_ioumat
from data import MultiClassData, MultiLabelData, DetectionData

class Evaluator(object):
    def __init__(self, annotations, iou_threshold=.1):
        self.annotations = annotations
        self.iou_threshold = iou_threshold
        self.label2class = self.annotations.label2class
        self.class2label = {y:x for x,y in self.label2class.items()}
        self.init_hook()

    def init_hook(self, *args):
        """
        function executed in the end of the __init__ function,
        can be used to do extra things for derived classes by overwriting it
        """
        pass

    def __str__(self):
        return "Unknown Metric"

    @property
    def image_names(self):
        return self.annotations.image_names

    @property
    def class_names(self):
        return list(self.label2class.values())

    def compute_metric(self, predictions):
        """
        Compute metric given predictions.
        Arguments:
            predictions: DataManager object, same source from the annotations
        Return:
            Any metric that set in config
        """
        raise NotImplementedError

    def _compute_instance_score(self, distribution, class_names):
        """ compute instance score given classes """
        assert len(list(class_names)) == len(set(class_names)), "class_names: '{}' containes duplicated class".format(class_names)
        if distribution is not None:
            class2label = self.class2label
            score = sum(distribution[class2label[cls_name]] for cls_name in class_names)
            return score
        else:
            return len(class_names) / (len(self.label2class)+1)