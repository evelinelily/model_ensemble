#
# Code for Hybrid Model Ensemble: Detection + Multi-Label -> Detection
#
from parameter import HybridParam
from data import ImageObject, InstanceObject, DetectionData, MultiLabelData
from me_utils import compute_ioumat
import numpy as np
import copy
from .base import Ensembler

class HybridEnsembler(Ensembler):
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
        super(HybridEnsembler, self).__init__([model1_pred, model2_pred])
        assert isinstance(model1_pred, DetectionData), "Model1 prediction must be a detection predictions"
        assert isinstance(model2_pred, MultiLabelData), "Model2 prediction must be a multilabel predictions"
        assert isinstance(param, HybridParam), "param must be a Detection parameter object"
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
        assert image_pred2.instance_num == 1, "image_pred2 is a multi-label prediction image object, must contain 1 and only 1 instance"
        dist_multilabel = image_pred2.instance_object_list[0].distribution

        ## 1) deal with overlapped predictions
        instance_object_list = list()
        for instance_detection in image_pred1.instance_object_list:
            # fuse roi box
            box = copy.copy(instance_detection.box)
            # fuse distribution
            distribution = instance_detection.distribution*self.param.dist_weights + dist_multilabel*(1.-self.param.dist_weights)
            distribution = distribution / max(distribution.sum(), 1e-8)   # distribution normalize to 1
            instance_object_list.append(InstanceObject(box=box, distribution=distribution))

        ## 2) form the fused results
        fused_image_object = ImageObject(image_pred1.image_name, instance_object_list)

        return fused_image_object