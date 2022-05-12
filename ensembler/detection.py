#
# Code for Model Ensemble
#
from parameter import DetectionParam
from data import ImageObject, InstanceObject, DetectionData
from me_utils import compute_ioumat
import numpy as np
import copy
from .base import Ensembler


class DetectionEnsembler(Ensembler):

    def __init__(self, param, model_predictions):
        """ 
        Initialize ensembler
        Arguments:
            param: parameter object for this ensembler
            predictions: a list of DataManager objects, represents DNN outputs from multiple models to be ensembled
        """
        assert isinstance(model_predictions, list) or isinstance(model_predictions, tuple)
        assert len(model_predictions) == 2, "Detection Ensemble only support two-model ensemble at this point"
        model1_pred, model2_pred = model_predictions
        super(DetectionEnsembler, self).__init__([model1_pred, model2_pred])
        assert isinstance(model1_pred, DetectionData), "Model1 prediction must be a detection predictions"
        assert isinstance(model2_pred, DetectionData), "Model2 prediction must be a detection predictions"
        assert isinstance(param, DetectionParam), "param must be a Detection parameter object"
        self.param = param
        self.param.initialize_weights(num_classes=model1_pred.class_num)

    def cast_fused_pred(self, data_obj):
        """ Cast the fused results to the right DataManager type """
        data_obj.__class__ = DetectionData
        return data_obj

    def fuse_image_preds(self, image_predictions):
        """
        Carry out model ensemble on current image with ensemble parameter self.param
        Arguments:
            image_predictions: list of ImageObject instance that record predictions of different model on a single image
        Return:
            a new ImageObject instance after the model ensemble
        """
        assert len(image_predictions) == 2, "Detection Ensemble only support two-model ensemble at this point"
        return self._fuse_image_preds(image_predictions[0], image_predictions[1])

    def _fuse_image_preds(self, image_pred1, image_pred2):
        """
        Carry out model ensemble on current image with ensemble parameter self.param
        Arguments:
            image_pred1/image_pred2: ImageObject instance that records predictions on single image
        Return:
            a new ImageObject instance after the model ensemble
        """
        assert image_pred1.image_name == image_pred2.image_name
        iou_mat = compute_ioumat(image_pred1.xyxys, image_pred2.xyxys)

        ## 1) deal with overlapped predictions
        overlap_ids1, overlap_ids2 = np.where(iou_mat > self.param.iou_threshold)
        instance_object_list = list()
        for id1, id2 in zip(overlap_ids1, overlap_ids2):
            instance_object1 = image_pred1.instance_object_list[id1]
            instance_object2 = image_pred2.instance_object_list[id2]
            # fuse roi box
            box = tuple(b1 * self.param.roi_weight + b2 * (1. - self.param.roi_weight)
                        for b1, b2 in zip(instance_object1.box, instance_object2.box))
            # fuse distribution
            distribution = instance_object1.distribution * np.array(
                self.param.dist_weights) + instance_object2.distribution * (1. - np.array(self.param.dist_weights))
            distribution = distribution / max(distribution.sum(), 1e-8)  # distribution normalize to 1
            instance_object_list.append(InstanceObject(box=box, distribution=distribution))

        ## 2) deal with lonely (unoverlapped) predictions
        # 2.1) lonely predictions from detector1
        for ix in range(image_pred1.instance_num):
            if not np.any(iou_mat[ix, :]):  # find lonely prediction
                assert ix not in overlap_ids1
                instance_object = image_pred1.instance_object_list[ix]
                box = instance_object.box
                fg_distribution = instance_object.distribution[1:] * self.param.lonely_fg_weight1
                bg_distribution = 1. - fg_distribution.sum()
                instance_object_list.append(
                    InstanceObject(box=box, distribution=np.hstack([bg_distribution, fg_distribution])))
        # 2.2) lonely predictions from detector2
        for ix in range(image_pred2.instance_num):
            if not np.any(iou_mat[:, ix]):  # find lonely prediction
                assert ix not in overlap_ids2
                instance_object = image_pred2.instance_object_list[ix]
                box = instance_object.box
                fg_distribution = instance_object.distribution[1:] * self.param.lonely_fg_weight2
                bg_distribution = 1. - fg_distribution.sum()
                instance_object_list.append(
                    InstanceObject(box=box, distribution=np.hstack([bg_distribution, fg_distribution])))

        ## 3) form the fused results
        fused_image_object = ImageObject(image_pred1.image_name, instance_object_list)
        fused_image_object.nms()  # run NMS to reduce duplications

        return fused_image_object