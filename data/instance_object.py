#
# Instance Object Class: managing individual instance
#
import numpy as np


class InstanceObject(object):

    def __init__(self, box, distribution):
        assert isinstance(box, tuple)
        assert isinstance(distribution, np.ndarray)
        self.box = box  # [Xmin, Ymin, Xmax, Ymax]
        self.distribution = distribution  # class distribution: fg, class1, class2, ...

    @property
    def area(self):
        return float(self.box[2] - self.box[0]) * float(self.box[3] - self.box[1])

    @property
    def fg_class_id(self):
        """ return the fg class id with maximal score """
        return np.argmax(self.distribution[1:]) + 1

    def json_format(self):
        x1, y1, x2, y2 = self.box
        return {
            "image_id": None,
            "instance_bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "instance_scores": [float(x) for x in self.distribution]
        }

    def dump(self):
        print("box: {}".format(self.box))
        print("distribution: {}".format(self.distribution))