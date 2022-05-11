#
# Ensembler to fuse two classifier
#
from parameter import MultiClassParam, MultiLabelParam, DetectionParam
from data import ImageObject, InstanceObject, MultiClassData, MultiLabelData
from me_utils import compute_ioumat
import numpy as np
import copy
from .base import Ensembler

class ClassificationEnsembler(Ensembler):
    param = None

    def normalize_distribution(self, distribution_unormalized):
        """ whether normalize sum(distribution) to 1 after fusion """
        raise NotImplementedError

    def fuse_image_preds(self, image_predictions):
        """
        Carry out model ensemble on current image with ensemble parameter self.param
        Arguments:
            image_predictions: list of ImageObject instance that record predictions of different model on a single image
        Return:
            a new ImageObject instance after the model ensemble
        """
        assert len(image_predictions) > 1
        assert len(image_predictions) == self.param.num_models, "#model inputs mismatch with the parameter: {} vs {}".format(len(image_predictions), self.param.num_models)
        image_pred0 = image_predictions[0]
        fused_distribution = None
        for iModel, image_pred in enumerate(image_predictions):
            assert image_pred.image_name == image_pred0.image_name
            assert image_pred.instance_num == 1

            instance_object = image_pred.instance_object_list[0]
            box = instance_object.box  # special boxes
            distribution = instance_object.distribution
            assert len(distribution) == self.param.num_classes

            # fuse distribution
            class_weights = self.param[iModel]
            assert len(class_weights) == len(distribution)
            if fused_distribution is not None:
                fused_distribution += class_weights * distribution
            else:
                fused_distribution  = class_weights * distribution

        distribution = self.normalize_distribution(distribution)
        instance_object_fused = InstanceObject(box=box, distribution=fused_distribution)
        fused_image_object = ImageObject(image_pred0.image_name, [instance_object_fused])

        return fused_image_object


class MultiClassEnsembler(ClassificationEnsembler):
    def __init__(self, param, model_predictions):
        """ 
        Initialize ensembler
        Arguments:
            param: parameter object for this ensembler
            predictions: a list of DataManager objects, represents DNN outputs from multiple models to be ensembled
        """
        assert all([isinstance(model_pred, MultiClassData) for model_pred in model_predictions]), "All model prediction must be a MultiClass predictions"
        assert isinstance(param, MultiClassParam), "param must be a MultiClass parameter object"

        super(MultiClassEnsembler, self).__init__(model_predictions)
        self.param = param
        self.param.initialize_weights(  # classification model only
            num_classes=model_predictions[0].class_num,
            num_models=len(model_predictions)
        )

    def normalize_distribution(self, distribution_unormalized):
        """ whether normalize sum(distribution) to 1 after fusion """
        return distribution_unormalized / max(distribution_unormalized.sum(), 1e-8)

    def cast_fused_pred(self, data_obj):
        """ Cast the fused results to the right DataManager type """
        data_obj.__class__ = MultiClassData
        return data_obj


class MultiLabelEnsembler(ClassificationEnsembler):
    def __init__(self, param, model_predictions):
        """ 
        Initialize ensembler
        Arguments:
            param: parameter object for this ensembler
            predictions: a list of DataManager objects, represents DNN outputs from multiple models to be ensembled
        """
        assert all([isinstance(model_pred, MultiLabelData) for model_pred in model_predictions]), "All model prediction must be a MultiLabel predictions"
        assert isinstance(param, MultiLabelParam), "param must be a MultiLabel parameter object"

        super(MultiLabelEnsembler, self).__init__(model_predictions)
        self.param = param
        self.param.initialize_weights(  # classification model only
            num_classes=model_predictions[0].class_num,
            num_models=len(model_predictions)
        )

    def normalize_distribution(self, distribution_unormalized):
        """ whether normalize sum(distribution) to 1 after fusion """
        return distribution_unormalized

    def cast_fused_pred(self, data_obj):
        """ Cast the fused results to the right DataManager type """
        data_obj.__class__ = MultiLabelData
        return data_obj