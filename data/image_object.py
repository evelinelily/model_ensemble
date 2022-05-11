#
# Image Object Class: managing image and instance objects
#
import numpy as np

import os, sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from me_utils import nms, compute_ioumat


class ImageObject(object):

    def __init__(self, image_name, instance_object_list):
        assert isinstance(instance_object_list, list)
        self.image_name = image_name
        self.instance_object_list = instance_object_list

    @property
    def instance_num(self):
        return len(self.instance_object_list)

    @property
    def xyxys(self):
        """ return bboxes as xyxys (np array) in default order """
        if self.instance_num == 0: return []
        boxes = np.vstack([x.box for x in self.instance_object_list])
        assert boxes.ndim == 2 and boxes.shape[1] == 4
        return boxes

    @property
    def dists(self):
        """ return stacked distributions ([-1, #num_classes]) in default order """
        if self.instance_num == 0: return []
        dists = np.vstack([x.distribution for x in self.instance_object_list])
        return dists

    def json_format(self):
        return {
            "image_id": None,
            "image_name": self.image_name,
        }

    def dump(self):
        print("image_name: {}, {} instances".format(self.image_name, self.instance_num))

    def nms(self, iou_threshold=.3):
        """ Run nms in place """
        if self.instance_num == 0:
            return

        all_xyxys = self.xyxys
        all_dists = self.dists
        num_fg_classes = all_dists.shape[1] - 1

        outstanding_fg_class_ids = np.argmax(all_dists[:, 1:], axis=1) + 1

        nms_selected_ids = list()
        for class_id in range(1, num_fg_classes + 1):
            inst_ids = np.where(outstanding_fg_class_ids == class_id)[0]
            xyxys = all_xyxys[inst_ids, :]
            dists = all_dists[inst_ids, :]
            scores = dists[:, class_id]
            left_ids = nms(xyxys, scores, overlapThresh=iou_threshold)
            nms_selected_ids.extend(list(inst_ids[left_ids]))

        # remove instances rejected by nms algorithm from instance_object_list
        self.instance_object_list = [self.instance_object_list[idx] for idx in nms_selected_ids]

    def equal_roi(self, obj):
        """ Check if the input image object has exactly the same instance (in roi) """
        if self.instance_num != obj.instance_num:
            return False

        if self.instance_num == 0:
            return True

        ioumat = compute_ioumat(self.xyxys, obj.xyxys)
        ioumat_overlap = np.zeros_like(ioumat)
        ioumat_overlap[ioumat == 1.] = 1

        return np.all(ioumat_overlap.sum(axis=0) == 1) and np.all(ioumat_overlap.sum(axis=1) == 1)