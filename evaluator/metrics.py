from .base import Evaluator
from .classification import MultiClassEvaluator, MultiLabelEvaluator
from .detection import DetectionEvaluator
from data import MultiLabelData, MultiClassData, DetectionData

class ApEvaluator(Evaluator):
    """ Evaluator that evaluate AP metrics """
    def compute_ap(self, predictions, config):
        raise NotImplementedError

    def select_evaluator(self, predictions):
        """ 
        Use the data type of predictions and annotations to decide with evaluator (multiclass/multilabel/detection)
        to use when computing AP
        """
        if None: pass
        elif isinstance(predictions, MultiClassData):
            self.__class__ = MultiClassEvaluator
        elif isinstance(predictions, MultiLabelData):
            self.__class__ = MultiLabelEvaluator
        elif isinstance(predictions, DetectionData):
            self.__class__ = DetectionEvaluator
        else:
            raise TypeError

class NgApEvaluator(ApEvaluator):
    def __str__(self):
        return "NG AP"

    def compute_metric(self, predictions):
        """
        Compute NG AP
        Arguments:
            predictions: DataManager object, same source from the annotations
        Return:
            Any metric that set in config
        """
        assert isinstance(predictions, DetectionData), "This metric can only be used in detection evaluation"
        self.select_evaluator(predictions)  # select the right evaluator to compute AP (temporarially)

        config = {'iou_threshold': self.iou_threshold, 'class_names': self.class_names}
        ap = self.compute_ap(predictions=predictions, config=config)

        self.__class__ = NgApEvaluator      # set back to the class itself
        return ap


class mApEvaluator(ApEvaluator):
    def __str__(self):
        return "mAP"

    def compute_metric(self, predictions):
        """
        Compute mean AP (mAP)
        Arguments:
            predictions: DataManager object, same source from the annotations
        Return:
            Any metric that set in config
        """
        self.select_evaluator(predictions)  # select the right evaluator to compute AP (temporarially)

        aps = list()
        for class_name in self.class_names:
            config = {'iou_threshold': self.iou_threshold, 'class_names': [class_name]}
            ap = self.compute_ap(predictions=predictions, config=config)
            aps.append(ap)
        mAP = sum(aps) / len(aps)

        self.__class__ = mApEvaluator      # set back to the class itself
        return mAP


class CustomizedmApEvaluator(object):
    @classmethod
    def set_classnames(clf, class_names):
        assert isinstance(class_names, list) or isinstance(class_names, tuple), "class_names must be a tuple or list"

        class NewEvaluator(ApEvaluator):
            def __str__(self):
                return "mAP of given classes '{}'".format(class_names)

            def init_hook(self, *args):
                """
                function executed in the end of the __init__ function,
                can be used to do extra things for derived classes by overwriting it
                """
                old_classnames = super(NewEvaluator, self).class_names
                assert set(class_names).issubset(set(old_classnames)), "Unknown class name in the input class_names"

            @property
            def class_names(self):
                return class_names

            def compute_metric(self, predictions):
                """
                Compute mean AP (mAP)
                Arguments:
                    predictions: DataManager object, same source from the annotations
                Return:
                    Any metric that set in config
                """
                self.select_evaluator(predictions)  # select the right evaluator to compute AP (temporarially)

                aps = list()
                for class_name in class_names:
                    config = {'iou_threshold': self.iou_threshold, 'class_names': [class_name]}
                    ap = self.compute_ap(predictions=predictions, config=config)
                    aps.append(ap)
                mAP = sum(aps) / len(aps)

                self.__class__ = NewEvaluator      # set back to the class itself
                return mAP

        return NewEvaluator


class CustomizedApEvaluator(object):
    @classmethod
    def set_classnames(clf, class_names):
        assert isinstance(class_names, list) or isinstance(class_names, tuple), "class_names must be a tuple or list"

        class NewEvaluator(ApEvaluator):
            def __str__(self):
                return "AP of class group: '{}'".format(class_names)

            def init_hook(self, *args):
                """
                function executed in the end of the __init__ function,
                can be used to do extra things for derived classes by overwriting it
                """
                old_classnames = super(NewEvaluator, self).class_names
                assert set(class_names).issubset(set(old_classnames)), "Unknown class name in the input class_names"

            @property
            def class_names(self):
                return class_names

            def compute_metric(self, predictions):
                """
                Compute AP
                Arguments:
                    predictions: DataManager object, same source from the annotations
                Return:
                    Any metric that set in config
                """
                self.select_evaluator(predictions)  # select the right evaluator to compute AP (temporarially)

                config = {'iou_threshold': self.iou_threshold, 'class_names': class_names}
                ap = self.compute_ap(predictions=predictions, config=config)

                self.__class__ = NewEvaluator      # set back to the class itself
                return ap

        return NewEvaluator

class PrecisionEvaluator(object):
    @classmethod
    def config(clf, class_names, recall_thresh):
        assert recall_thresh >= .0 and recall_thresh <= 1.
        assert isinstance(class_names, list) or isinstance(class_names, tuple), "class_names must be a tuple or list"

        class NewEvaluator(Evaluator):
            """ Compute precision recall at 100 """
            def compute_pr_at_recall(self, predictions, config):
                raise NotImplementedError

            def __str__(self):
                return "Precision at Recall={}".format(recall_thresh)

            def compute_metric(self, predictions):
                """
                Compute mean AP (mAP)
                Arguments:
                    predictions: DataManager object, same source from the annotations
                Return:
                    Any metric that set in config
                """
                config = {'recall_thresh': recall_thresh, 'class_names': class_names}
                if None: pass
                elif isinstance(predictions, MultiClassData):
                    self.__class__ = MultiClassEvaluator
                elif isinstance(predictions, MultiLabelData):
                    self.__class__ = MultiLabelEvaluator
                elif isinstance(predictions, DetectionData):
                    # when compute pr@rec=x, the evaluation for detectors devolve from the instance-level metric to image-level metric
                    annotations_multilabel = MultiLabelData.from_DetectionData(self.annotations)
                    predictions_multilabel = MultiLabelData.from_DetectionData(predictions)
                    ev = MultiLabelEvaluator(annotations_multilabel)
                    precision = ev.compute_pr_at_recall(predictions_multilabel, config)
                    return precision
                else:
                    raise TypeError

                precision = self.compute_pr_at_recall(predictions, config)

                self.__class__ = NewEvaluator      # set back to the class itself
                return precision

        return NewEvaluator


class RecallEvaluator(object):
    @classmethod
    def config(clf, class_names, precision_thresh):
        assert precision_thresh >= .0 and precision_thresh <= 1.
        assert isinstance(class_names, list) or isinstance(class_names, tuple), "class_names must be a tuple or list"

        class NewEvaluator(Evaluator):
            """ Compute recall when precision at xxx """
            def compute_rec_at_prec(self, predictions, config):
                raise NotImplementedError

            def __str__(self):
                return "Recall at Precision={}".format(precision_thresh)

            def compute_metric(self, predictions):
                """
                Compute mean AP (mAP)
                Arguments:
                    predictions: DataManager object, same source from the annotations
                Return:
                    Any metric that set in config
                """
                config = {'precision_thresh': precision_thresh, 'class_names': class_names}
                if None: pass
                elif isinstance(predictions, MultiClassData):
                    self.__class__ = MultiClassEvaluator
                elif isinstance(predictions, MultiLabelData):
                    self.__class__ = MultiLabelEvaluator
                elif isinstance(predictions, DetectionData):
                    # when compute pr@rec=x, the evaluation for detectors devolve from the instance-level metric to image-level metric
                    annotations_multilabel = MultiLabelData.from_DetectionData(self.annotations)
                    predictions_multilabel = MultiLabelData.from_DetectionData(predictions)
                    ev = MultiLabelEvaluator(annotations_multilabel)
                    precision = ev.compute_rec_at_prec(predictions_multilabel, config)
                    return precision
                else:
                    raise TypeError

                precision = self.compute_rec_at_prec(predictions, config)

                self.__class__ = NewEvaluator      # set back to the class itself
                return precision

        return NewEvaluator