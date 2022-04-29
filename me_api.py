#
# Model Ensemble API
#

import os, sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model_ensemble'))
from data import ImageObject, InstanceObject
from evaluator import mApEvaluator, NgApEvaluator, PrecisionEvaluator, CustomizedmApEvaluator, CustomizedApEvaluator, RecallEvaluator
from evaluator import Evaluator as EvaluatorBase
from optimizer import RandomSearchOptimizer, PassiveOptimizer, BayesianOptimizer, GeneticOptimizer
import numpy as np
import copy

all_metrics = {
    'mAP': mApEvaluator,
    'ngAP': NgApEvaluator,
    'classmAP': CustomizedmApEvaluator,
    'classAP': CustomizedApEvaluator,
    'prec@rec': PrecisionEvaluator,
    'rec@prec': RecallEvaluator
}
all_optimizers = {
    'bayesian': BayesianOptimizer,
    'passive': PassiveOptimizer,
    'random': RandomSearchOptimizer,
    'genetic': GeneticOptimizer
}


### 模型融合基类，分类模型融合类、检测模型融合类从该类继承 ###
class ModelEnsembleBase(object):
    """
    Model Ensemble API Class
    """

    def encode_prediction(self, prediction, alias='image-name'):
        """
        Encode single image prediction to ImageObject instance
        The input prediction is a tuple or list
        """
        raise NotImplementedError

    def decode_prediction(self, image_object):
        """
        Convert ImageObject instance to interpretable prediction
        The output prediction is a tuple or list
        """
        raise NotImplementedError

    def initialize_parameter(self, parameter):
        """
        Initialize ME hyper-parameters
        Arguments:
            param: ParameterClass instance that keep the track of ME hyper-parameters
        """
        assert isinstance(parameter, self.Parameter)
        self.param = parameter
        dummy_data1 = self.InputDataClass1({}, {})
        dummy_data2 = self.InputDataClass2({}, {})
        self.ensembler = self.Ensembler(parameter, [dummy_data1, dummy_data2])

    def fuse(self, prediction1, prediction2, *args, **kwargs):
        """
        Run model ensemble with predictions from two models on a single image
        Arguments:
            prediction1/2: predictions from model1/2 on a single image;
                            for detection predictions, they are a list/tuple of list/tuple, where
                            prediction[i] is a bbox prediction on image: ([bg_prob, cls1_prob, cls2_prob, ...], [Xmin, Ymin, Xmax, Ymax])
        Return:
            prediction_fused: fused predictions that has the same format as the input predictions
        """
        image_object_list = [self.encode_prediction(prediction1), self.encode_prediction(prediction2)]
        # deal with case when we ensemble multiply models
        assert len(args) == 0 or len(kwargs) == 0
        if len(args) > 0:
            image_object_list.extend(list(map(self.encode_prediction, args)))
        if len(kwargs) > 0:
            assert len(kwargs) < 100, "Error: too many models to be ensembled: {}".format(len(kwargs) + 2)
            for ix in range(3, len(kwargs) + 3):
                arg = 'prediction{}'.format(ix)
                assert arg in kwargs, "expecting argument: {}".format(arg)
                image_object_list.append(self.encode_prediction(kwargs[arg]))
        # fuse predictions
        image_object_fused = self.ensembler.fuse_image_preds(image_object_list)
        prediction_fused = self.decode_prediction(image_object_fused)
        return prediction_fused

    def fake_class_dict(self, predictions):
        """ fake a class dict (label2class) from the predictions (by observing the lenghth of the distribution) """
        raise NotImplementedError

    def initialize_optimizer(self, predictions1, predictions2, ground_truth, **kwargs):
        """
        Initialize optimizer with validation data.
        Arguments:
            predictions1/2: list of predictions, of which the definition can be find in the argument description in self.fuse()
            ground_truth: same format as predictions1/2; the distribution must be one-hot distribution 
        """
        predictions_list = [predictions1, predictions2]
        if len(kwargs) > 0:
            assert len(kwargs) < 100, "Error: too many models to be ensembled: {}".format(len(kwargs) + 2)
            for ix in range(3, len(kwargs) + 3):
                arg = 'predictions{}'.format(ix)
                assert arg in kwargs, "expecting argument: {}".format(arg)
                predictions_list.append(kwargs[arg])

        # check image length
        num_images = len(ground_truth)
        assert all([len(x) == num_images
                    for x in predictions_list]), "#samples of predictions and the ground truths unmatch"

        # convert parse input data
        image_object_dict_predictions = [dict()
                                         for _ in range(len(predictions_list))]  # image_object_dict for predictions
        image_object_dict_gth = dict()  # image_object_dict for ground truth
        for iImage, gth in enumerate(ground_truth):
            dummy_name = 'dummy-{}'.format(iImage)
            # load groundtruth
            image_object_dict_gth[dummy_name] = self.encode_prediction(gth, alias=dummy_name)
            if image_object_dict_gth[dummy_name].instance_num > 0:
                dists = image_object_dict_gth[dummy_name].dists
                assert np.all(np.logical_or(dists == 0, dists == 1)), "Groundtruths is not onehot: {}".format(dists)
            # load predictions
            for iPred, predictions in enumerate(predictions_list):
                pred = predictions[iImage]
                image_object_dict_predictions[iPred][dummy_name] = self.encode_prediction(pred, alias=dummy_name)

        # create label2class
        label2class1, label2class2, label2class_out = self.fake_class_dict(ground_truth)
        # load gth
        self.groundtruths_val = self.OutputDataClass(image_object_dict_gth, copy.copy(label2class_out))
        # set data types of predictions>3 as predictions1
        label2class_list = [label2class1, label2class2] + [label2class1] * (len(predictions_list) - 2)
        InputDataClass_list = [self.InputDataClass1, self.InputDataClass2
                               ] + [self.InputDataClass1] * (len(predictions_list) - 2)
        assert len(image_object_dict_predictions) == len(label2class_list) == len(InputDataClass_list)
        # load predictions
        self.predictions_val = list()
        for image_object_dict, label2class, InputDataClass in zip(image_object_dict_predictions, label2class_list,
                                                                  InputDataClass_list):
            self.predictions_val.append(InputDataClass(image_object_dict, copy.copy(label2class)))

    def initialize_evaluator(self, predictions1, predictions2, ground_truth, **kwargs):
        """
        Initialize optimizer with validation data.
        Arguments:
            predictions1/2: list of predictions, of which the definition can be find in the argument description in self.fuse()
            ground_truth: same format as predictions1/2; the distribution must be one-hot distribution 
        """
        predictions_list = [predictions1, predictions2]
        if len(kwargs) > 0:
            assert len(kwargs) < 100, "Error: too many models to be ensembled: {}".format(len(kwargs) + 2)
            for ix in range(3, len(kwargs) + 3):
                arg = 'predictions{}'.format(ix)
                assert arg in kwargs, "expecting argument: {}".format(arg)
                predictions_list.append(kwargs[arg])

        # check image length
        num_images = len(ground_truth)
        assert all([len(x) == num_images
                    for x in predictions_list]), "#samples of predictions and the ground truths unmatch"

        # convert parse input data
        image_object_dict_predictions = [dict()
                                         for _ in range(len(predictions_list))]  # image_object_dict for predictions
        image_object_dict_gth = dict()  # image_object_dict for ground truth
        for iImage, gth in enumerate(ground_truth):
            dummy_name = 'dummy-{}'.format(iImage)
            # load groundtruth
            image_object_dict_gth[dummy_name] = self.encode_prediction(gth, alias=dummy_name)
            if image_object_dict_gth[dummy_name].instance_num > 0:
                dists = image_object_dict_gth[dummy_name].dists
                assert np.all(np.logical_or(dists == 0, dists == 1)), "Groundtruths is not onehot: {}".format(dists)
            # load predictions
            for iPred, predictions in enumerate(predictions_list):
                pred = predictions[iImage]
                image_object_dict_predictions[iPred][dummy_name] = self.encode_prediction(pred, alias=dummy_name)

        # create label2class
        label2class1, label2class2, label2class_out = self.fake_class_dict(ground_truth)
        # load gth
        self.groundtruths_test = self.OutputDataClass(image_object_dict_gth, copy.copy(label2class_out))
        # set data types of predictions>3 as predictions1
        label2class_list = [label2class1, label2class2] + [label2class1] * (len(predictions_list) - 2)
        InputDataClass_list = [self.InputDataClass1, self.InputDataClass2
                               ] + [self.InputDataClass1] * (len(predictions_list) - 2)
        assert len(image_object_dict_predictions) == len(label2class_list) == len(InputDataClass_list)
        # load predictions
        self.predictions_test = list()
        for image_object_dict, label2class, InputDataClass in zip(image_object_dict_predictions, label2class_list,
                                                                  InputDataClass_list):
            self.predictions_test.append(InputDataClass(image_object_dict, copy.copy(label2class)))

    def _setup_metric(self, metric='mAP', **kargs):
        """
        Set up metric.
        Arguments:
            metric: a string object or callable function (metric callback)
            kargs: when metric in ['prec@rec', 'classAP', 'rec@prec'], some arguments such as 'class_names' for these metrics needs to be set
        Return:
            A Evaluator class bond with the given metric
        """
        assert metric in all_metrics or callable(metric)

        # if metric is a metric callback function, which takes predictions and ground_truth as inputs, and outputs a score (customized metric)
        if callable(metric):
            decode_prediction_func = self.decode_prediction

            class Evaluator(EvaluatorBase):

                def compute_metric(self, predictions):
                    assert isinstance(predictions, self.OutputDataClass)
                    # decode predictions and annotations to plain data strcture
                    predictions_decoded, annotations_decoded = list(), list()
                    for img_name in predictions.image_names:
                        pred, gth = predictions[img_name], self.annotations[img_name]
                        predictions_decoded.append(decode_prediction_func(pred))
                        annotations_decoded.append(decode_prediction_func(gth))
                    return metric(predictions_decoded, annotations_decoded)
        else:
            Evaluator = all_metrics[metric]
            if None: pass
            elif metric == 'classAP':
                assert 'class_names' in kargs, "classAP needs to set argument 'class_names'"
                Evaluator = Evaluator.set_classnames(class_names=kargs['class_names'])
            elif metric == 'prec@rec':
                assert 'class_names' in kargs, "prec@rec needs to set argument 'class_names'"
                assert 'recall_threshold' in kargs, "prec@rec needs to set argument 'recall_threshold'"
                Evaluator = Evaluator.config(class_names=kargs['class_names'], recall_thresh=kargs['recall_threshold'])
            elif metric == 'rec@prec':
                assert 'class_names' in kargs, "rec@prec needs to set argument 'class_names'"
                assert 'precision_threshold' in kargs, "rec@prec needs to set argument 'precision_threshold'"
                Evaluator = Evaluator.config(class_names=kargs['class_names'],
                                             precision_thresh=kargs['precision_threshold'])
            else:
                pass

        return Evaluator

    def maximize(self, metric='mAP', optimizer='baysian', **kargs):
        """
        Maximize the given metric with the given optimizer
        Arguments:
            metric: a string object or callable function (metric callback)
            optimizer: a string object
            kargs: when metric in ['prec@rec', 'classAP', 'rec@prec'], some arguments such as 'class_names' for these metrics needs to be set
        """
        Evaluator = self._setup_metric(metric=metric, **kargs)

        # set iou_threshold
        if 'iou_threshold' in kargs:
            iou_threshold = kargs['iou_threshold']
            assert 0 <= iou_threshold <= 1.
        else:
            iou_threshold = 0.1

        # run optimization
        assert optimizer in all_optimizers
        Optimizer = all_optimizers[optimizer]
        ensembler = self.Ensembler(self.Parameter(), model_predictions=self.predictions_val)
        evaluator = Evaluator(annotations=self.groundtruths_val, iou_threshold=iou_threshold)
        optim = Optimizer(ensembler, evaluator)
        param_opt = optim.maximize()

        return param_opt

    def eval(self, metric='mAP', **kargs):
        """ Evaluate current data with current parameters """
        # check initialization
        assert hasattr(self, 'groundtruths_test'), "Run initialize_evaluator() before running eval()"
        assert hasattr(self, 'param'), "Run initialize_parameter() before running eval()"

        # set up metric
        Evaluator = self._setup_metric(metric=metric, **kargs)

        # set iou_threshold
        if 'iou_threshold' in kargs:
            iou_threshold = kargs['iou_threshold']
            assert 0 <= iou_threshold <= 1.
        else:
            iou_threshold = 0.1
        ev = Evaluator(annotations=self.groundtruths_test, iou_threshold=iou_threshold)

        # dump final results
        metrics_before_me = list()
        for predictions in self.predictions_test:
            try:
                metric = ev.compute_metric(predictions)
            except:
                metric = 'NA.'
            metrics_before_me.append(metric)

        ev = Evaluator(annotations=self.groundtruths_test, iou_threshold=iou_threshold)
        ensemble_optimal = self.Ensembler(self.param, model_predictions=self.predictions_test)
        ensemble_default = self.Ensembler(self.Parameter(), model_predictions=self.predictions_test)
        fused_preds_optimal = ensemble_optimal.fuse()
        fused_preds_default = ensemble_default.fuse()
        metric_fused_optimal = ev.compute_metric(fused_preds_optimal)
        metric_fused_default = ev.compute_metric(fused_preds_default)
        print("Metric Name: {}".format(str(ev)))
        for ix, metric in enumerate(metrics_before_me):
            print("Model{}: {}".format(ix + 1, metric))
        print("Default Fused Model: {}".format(metric_fused_default))
        print("Optimal Fused Model: {}".format(metric_fused_optimal))


### 检测模型融合类 ###
from data import DetectionData
from parameter import DetectionParam
from ensembler import DetectionEnsembler


class DetectionModelEnsemble(ModelEnsembleBase):
    InputDataClass1 = DetectionData
    InputDataClass2 = DetectionData
    OutputDataClass = DetectionData
    Parameter = DetectionParam
    Ensembler = DetectionEnsembler

    def fake_class_dict(self, predictions):
        """ fake a class dict from the predictions (by observing the lenghth of the distribution) """
        num_fg_classes = 0
        for pred in predictions:
            if len(pred) > 0:
                num_fg_classes = len(pred[0][0]) - 1
                break
        assert num_fg_classes > 0, "Error: empty predictions and ground truths can't be used as data for hyper-parameter optimization"
        label2class = {ix + 1: 'class{}'.format(ix + 1) for ix in range(num_fg_classes)}
        return label2class, label2class, label2class

    def encode_prediction(self, prediction, alias='image-name'):
        """
        Encode single image prediction to ImageObject instance
        The input prediction is a tuple or list, for prediction[i]: ([bg_prob, cls1_prob, cls2_prob, ...], [Xmin, Ymin, Xmax, Ymax])
        """
        instance_object_list = [InstanceObject(box=tuple(box), distribution=np.array(dist)) for dist, box in prediction]
        image_object = ImageObject(image_name=alias, instance_object_list=instance_object_list)
        return image_object

    def decode_prediction(self, image_object):
        """
        Convert ImageObject instance to interpretable prediction
        The output prediction is a tuple or list, for prediction[i]: ([bg_prob, cls1_prob, cls2_prob, ...], [Xmin, Ymin, Xmax, Ymax])
        """
        dists, boxes = image_object.dists, image_object.xyxys
        preds = list()
        for ix in range(image_object.instance_num):
            preds.append((dists[ix, :].tolist(), boxes[ix, :].tolist()))
        return preds


### 分类模型融合类 ###
class ClassificationModelEnsemble(ModelEnsembleBase):

    def fake_class_dict(self, predictions):
        """ fake a class dict from the predictions (by observing the lenghth of the distribution) """
        pred0 = predictions[0]
        label2class = {ix: 'class{}'.format(ix) for ix in range(len(pred0))}
        return label2class, label2class, label2class

    def encode_prediction(self, prediction, alias='image-name'):
        """
        Encode single image prediction to ImageObject instance
        The input prediction is a tuple or list: [cls1_prob, cls2_prob, ...]
        """
        assert len(prediction) > 0
        instance_object_list = [InstanceObject(box=tuple(), distribution=np.array(prediction))]
        image_object = ImageObject(image_name=alias, instance_object_list=instance_object_list)
        return image_object

    def decode_prediction(self, image_object):
        """
        Convert ImageObject instance to interpretable prediction
        The output prediction is a tuple or list: [cls1_prob, cls2_prob, ...]
        """
        assert image_object.instance_num == 1
        pred = image_object.dists.flatten().tolist()
        return pred


### MultiClass分类融合类 ###
from data import MultiClassData
from parameter import MultiClassParam
from ensembler import MultiClassEnsembler


class MultiClassModelEnsemble(ClassificationModelEnsemble):
    InputDataClass1 = MultiClassData
    InputDataClass2 = MultiClassData
    OutputDataClass = MultiClassData
    Parameter = MultiClassParam
    Ensembler = MultiClassEnsembler


### MultiLabel分类融合类 ###
from data import MultiLabelData
from parameter import MultiLabelParam
from ensembler import MultiLabelEnsembler


class MultiLabelModelEnsemble(ClassificationModelEnsemble):
    InputDataClass1 = MultiLabelData
    InputDataClass2 = MultiLabelData
    OutputDataClass = MultiLabelData
    Parameter = MultiLabelParam
    Ensembler = MultiLabelEnsembler


### Hybrid融合类（目标检测+多标签分类） ###
from parameter import HybridParam
from ensembler import HybridEnsembler


class HybridModelEnsemble(DetectionModelEnsemble):
    InputDataClass1 = DetectionData
    InputDataClass2 = MultiLabelData
    OutputDataClass = DetectionData
    Parameter = HybridParam
    Ensembler = HybridEnsembler

    def fake_class_dict(self, detections):
        """ fake a class dict from the detections (by observing the lenghth of the distribution) """
        num_fg_classes = 0
        for pred in detections:
            if len(pred) > 0:
                num_fg_classes = len(pred[0][0]) - 1
                break
        assert num_fg_classes > 0, "Error: empty detections and ground truths can't be used as data for hyper-parameter optimization"
        label2class1 = {ix + 1: 'class{}'.format(ix + 1) for ix in range(num_fg_classes)}
        label2class2 = {ix: 'class{}'.format(ix) for ix in range(num_fg_classes + 1)}
        return label2class1, label2class2, label2class1

    def encode_prediction(self, prediction, alias='image-name'):
        """
        Encode single image prediction to ImageObject instance
        The input prediction is a tuple or list, for prediction[i]: ([bg_prob, cls1_prob, cls2_prob, ...], [Xmin, Ymin, Xmax, Ymax])
        """
        if len(prediction) == 0 or isinstance(prediction[0], tuple) or isinstance(prediction[0],
                                                                                  list):  # detection prediction format
            instance_object_list = [
                InstanceObject(box=tuple(box), distribution=np.array(dist)) for dist, box in prediction
            ]
        else:
            instance_object_list = [InstanceObject(box=tuple(), distribution=np.array(prediction))]
        image_object = ImageObject(image_name=alias, instance_object_list=instance_object_list)
        return image_object


### 指定融合模式接口函数 ###
def import_me_classes(mode):
    """
    指定融合模式，目前共支持4种："detection", "multi-class", "multi-label"，"hybrid"。

    """
    if None: pass
    elif mode == 'detection':
        return DetectionModelEnsemble, DetectionParam
    elif mode == 'multi-class':
        return MultiClassModelEnsemble, MultiClassParam
    elif mode == 'multi-label':
        return MultiLabelModelEnsemble, MultiLabelParam
    elif mode == 'hybrid':
        return HybridModelEnsemble, HybridParam
    else:
        raise NotImplementedError