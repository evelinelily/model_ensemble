#
# Code for Model Ensemble
#
from data import DataManager
from me_utils import compute_ioumat
import numpy as np
import copy
from multiprocessing import Pool

class Ensembler(object):
    def __init__(self, model_predictions):
        """ 
        Initialize ensembler
        Arguments:
            predictions: a list of DataManager objects, represents DNN outputs from multiple models to be ensembled
        """
        self.model_predictions = model_predictions
        assert isinstance(model_predictions, list) or isinstance(model_predictions, tuple)
        assert len(model_predictions) > 1, "You must feed in at least 1 model predictions"
        model_pred0 = model_predictions[0]
        for model_pred in model_predictions:
            assert isinstance(model_pred, DataManager)
            assert model_pred0.compatible_with(model_pred), "predictions models preds should be compatible with each others"

    def cast_fused_pred(self, data_obj):
        """ Cast the fused results to the right DataManager type """
        raise NotImplementedError

    def fuse_image_preds(self, image_predictions):
        """
        Carry out model ensemble on current image with ensemble parameter self.param
        Arguments:
            image_predictions: list of ImageObject instance that record predictions of different model on a single image
        Return:
            a new ImageObject instance after the model ensemble
        """
        raise NotImplementedError

    def fuse(self):
        """
        Carry out model ensemble for the whole dataset with ensemble parameter self.param
        Return: a DataManager object
        """
        model_pred0 = self.model_predictions[0]
        label2class = copy.deepcopy(model_pred0.label2class)
        fused_image_object_dict = dict()
        for image_name in model_pred0.image_names:
            model_image_pred = [model_pred[image_name] for model_pred in self.model_predictions]
            fused_image_pred = self.fuse_image_preds(image_predictions = model_image_pred)
            fused_image_object_dict[image_name] = fused_image_pred
        fused_preds = DataManager(image_object_dict=fused_image_object_dict, label2class=label2class)
        fused_preds = self.cast_fused_pred(fused_preds)
        return fused_preds