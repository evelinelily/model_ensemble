#
# Evaluator for Classification results
#
from data import MultiClassData, MultiLabelData
from me_utils import compute_r_at_p
from .base import Evaluator
from .classifier_eval import ClassifierEvalBinary
import numpy as np

class ClassificationEvaluator(Evaluator):
    def compute_ap(self, predictions, config):
        """ Compute AP given predictions and config """
        # compute AP
        y_true = list()
        y_score = list()
        for image_name in self.image_names:
            image_object_pred = predictions[image_name]
            image_object_anno = self.annotations[image_name]
            distribution_pred = image_object_pred.instance_object_list[0].distribution
            distribution_anno = image_object_anno.instance_object_list[0].distribution
            # get prediction and groundtruth
            gth_class_name = self.label2class[np.argmax(distribution_anno)]
            y_true.append(int(gth_class_name in config['class_names']))
            y_score.append(
                self._compute_instance_score(distribution_pred, config['class_names'])
            )
        ap = ClassifierEvalBinary.compute_ap(y_true=y_true, y_score=y_score)

        return ap

    def compute_pr_at_recall(self, predictions, config):
        """ Compute precision when recall = config['recall_thresh'] """
        # compute AP
        y_true = list()
        y_score = list()
        for image_name in self.image_names:
            image_object_pred = predictions[image_name]
            image_object_anno = self.annotations[image_name]
            distribution_pred = image_object_pred.instance_object_list[0].distribution
            distribution_anno = image_object_anno.instance_object_list[0].distribution
            # get prediction and groundtruth
            gth_class_name = self.label2class[np.argmax(distribution_anno)]
            y_true.append(int(gth_class_name in config['class_names']))
            y_score.append(
                self._compute_instance_score(distribution_pred, config['class_names'])
            )
        pr = ClassifierEvalBinary.compute_p_at_r(y_true=y_true, y_score=y_score, recall_thresh=config['recall_thresh'])

        return pr

    def compute_rec_at_precision(self, predictions, config):
        """ Compute precision when recall = config['recall_thresh'] """
        # compute AP
        y_true = list()
        y_score = list()
        for image_name in self.image_names:
            image_object_pred = predictions[image_name]
            image_object_anno = self.annotations[image_name]
            distribution_pred = image_object_pred.instance_object_list[0].distribution
            distribution_anno = image_object_anno.instance_object_list[0].distribution
            # get prediction and groundtruth
            gth_class_name = self.label2class[np.argmax(distribution_anno)]
            y_true.append(int(gth_class_name in config['class_names']))
            y_score.append(
                self._compute_instance_score(distribution_pred, config['class_names'])
            )
        rec = compute_r_at_p(y_true=y_true, y_score=y_score, precision_thresh=config['precision_thresh'])

        return rec


class MultiClassEvaluator(ClassificationEvaluator):
    def compute_ap(self, predictions, config):
        """ Compute AP given predictions and config """
        assert isinstance(predictions, MultiClassData), "predictions should be MultiClassData type."
        assert isinstance(self.annotations, MultiClassData), "annotations should be MultiClassData type."
        assert self.annotations.compatible_with(predictions), "predictions not match with annotations."
        assert 'class_names' in config
        return super(MultiClassEvaluator, self).compute_ap(predictions, config)

    def compute_pr_at_recall(self, predictions, config):
        """ Compute precision when recall = config['recall_thresh'] """
        assert isinstance(predictions, MultiClassData), "predictions should be MultiClassData type."
        assert isinstance(self.annotations, MultiClassData), "annotations should be MultiClassData type."
        assert self.annotations.compatible_with(predictions), "predictions not match with annotations."
        assert 'class_names' in config and 'recall_thresh' in config
        return super(MultiClassEvaluator, self).compute_pr_at_recall(predictions, config)

    def compute_rec_at_prec(self, predictions, config):
        """
        Compute recall when precision = config['precision_thresh']
        precision = precision_thresh is not always gaurrenteed, so we add a punishment term to the metric (to make it smaller)
        when it happens
        """
        assert isinstance(predictions, MultiClassData), "predictions should be MultiClassData type."
        assert isinstance(self.annotations, MultiClassData), "annotations should be MultiClassData type."
        assert self.annotations.compatible_with(predictions), "predictions not match with annotations."
        assert 'class_names' in config and 'precision_thresh' in config
        return super(MultiClassEvaluator, self).compute_rec_at_precision(predictions, config)

        

class MultiLabelEvaluator(ClassificationEvaluator):
    def compute_ap(self, predictions, config):
        """ Compute AP given predictions and config """
        assert isinstance(predictions, MultiLabelData), "predictions should be MultiLabelData type."
        assert isinstance(self.annotations, MultiLabelData), "annotations should be MultiLabelData type."
        assert self.annotations.compatible_with(predictions), "predictions not match with annotations."
        return super(MultiLabelEvaluator, self).compute_ap(predictions, config)

    def compute_pr_at_recall(self, predictions, config):
        """ Compute precision when recall = config['recall_thresh'] """
        assert isinstance(predictions, MultiLabelData), "predictions should be MultiLabelData type."
        assert isinstance(self.annotations, MultiLabelData), "annotations should be MultiLabelData type."
        assert self.annotations.compatible_with(predictions), "predictions not match with annotations."
        assert 'class_names' in config and 'recall_thresh' in config
        return super(MultiLabelEvaluator, self).compute_pr_at_recall(predictions, config)

    def compute_rec_at_prec(self, predictions, config):
        """
        Compute recall when precision = config['precision_thresh']
        precision = precision_thresh is not always gaurrenteed, so we add a punishment term to the metric (to make it smaller)
        when it happens
        """
        assert isinstance(predictions, MultiLabelData), "predictions should be MultiLabelData type."
        assert isinstance(self.annotations, MultiLabelData), "annotations should be MultiLabelData type."
        assert self.annotations.compatible_with(predictions), "predictions not match with annotations."
        assert 'class_names' in config and 'precision_thresh' in config
        return super(MultiLabelEvaluator, self).compute_rec_at_precision(predictions, config)

    def _compute_instance_score(self, distribution, class_names):
        """ compute instance score given classes """
        assert len(list(class_names)) == len(set(class_names)), "class_names: '{}' containes duplicated class".format(class_names)
        if distribution is not None:
            class2label = self.class2label
            score = max(distribution[class2label[cls_name]] for cls_name in class_names)
            return score
        else:
            return len(class_names) / (len(self.label2class)+1)