#
# enumerate configurations of all experiments
#
from data import MultiLabelData, MultiClassData, DetectionData
from ensembler import MultiClassEnsembler, MultiLabelEnsembler, DetectionEnsembler, HybridEnsembler
from parameter import MultiClassParam, MultiLabelParam, DetectionParam, HybridParam
from evaluator import mApEvaluator, NgApEvaluator, CustomizedmApEvaluator, PrecisionEvaluator, RecallEvaluator
from optimizer import RandomSearchOptimizer, PassiveOptimizer, BayesianOptimizer, GeneticOptimizer

def kangqiang_24l():
    val_preds1 = DetectionData.from_json('assets/kangqiang/24L/DIFF/val.json')
    val_preds2 = DetectionData.from_json('assets/kangqiang/24L/FPN/val.json')
    val_gths = DetectionData.from_json('assets/kangqiang/24L/gth/val.json')

    test_preds1 = DetectionData.from_json('assets/kangqiang/24L/DIFF/test.json')
    test_preds2 = DetectionData.from_json('assets/kangqiang/24L/FPN/test.json')
    test_gths = DetectionData.from_json('assets/kangqiang/24L/gth/test.json')

    config = {
        'val-preds1': val_preds1,
        'val-preds2': val_preds2,
        'val-gths': val_gths,
        'test-preds1': test_preds1,
        'test-preds2': test_preds2,
        'test-gths': test_gths,
        'ParamManager': DetectionParam,
        'Ensembler': DetectionEnsembler,
        'Evaluator': mApEvaluator,
        'Optimizer': GeneticOptimizer,
    }
    return config


def tianhe():
    val_preds1 = DetectionData.from_json('assets/tianhe/val-ssd.json')
    val_preds2 = DetectionData.from_json('assets/tianhe/val-yolo.json')
    val_gths = DetectionData.from_json('assets/tianhe/val-gths.json')

    test_preds1 = DetectionData.from_json('assets/tianhe/test-ssd.json')
    test_preds2 = DetectionData.from_json('assets/tianhe/test-yolo.json')
    test_gths = DetectionData.from_json('assets/tianhe/test-gths.json')

    config = {
        'val-preds1': val_preds1,
        'val-preds2': val_preds2,
        'val-gths': val_gths,
        'test-preds1': test_preds1,
        'test-preds2': test_preds2,
        'test-gths': test_gths,
        'ParamManager': DetectionParam,
        'Ensembler': DetectionEnsembler,
        'Evaluator': mApEvaluator,
        'Optimizer': BayesianOptimizer,
    }
    return config


def kangqiang_classifier_28l():
    val_preds1 = MultiClassData.from_pred_json('assets/kangqiang/classification/binary/28l/val1.json')
    val_preds2 = MultiClassData.from_pred_json('assets/kangqiang/classification/binary/28l/val2.json')
    val_gths = MultiClassData.from_gth_json('assets/kangqiang/classification/binary/28l/val1.json')

    test_preds1 = MultiClassData.from_pred_json('assets/kangqiang/classification/binary/28l/test1.json')
    test_preds2 = MultiClassData.from_pred_json('assets/kangqiang/classification/binary/28l/test2.json')
    test_gths = MultiClassData.from_gth_json('assets/kangqiang/classification/binary/28l/test1.json')

    config = {
        'val-preds1': val_preds1,
        'val-preds2': val_preds2,
        'val-gths': val_gths,
        'test-preds1': test_preds1,
        'test-preds2': test_preds2,
        'test-gths': test_gths,
        'ParamManager': MultiClassParam,
        'Ensembler': MultiClassEnsembler,
        'Evaluator': PrecisionEvaluator.config(recall_thresh=1., class_names=['NG']),
        'Optimizer': GeneticOptimizer,
    }
    return config

def kangqiang_classifier_40l():
    val_preds1 = MultiClassData.from_pred_json('assets/kangqiang/classification/binary/40l/val-16.json')
    val_preds2 = MultiClassData.from_pred_json('assets/kangqiang/classification/binary/40l/val-28.json')
    val_gths = MultiClassData.from_gth_json('assets/kangqiang/classification/binary/40l/val-16.json')

    test_preds1 = MultiClassData.from_pred_json('assets/kangqiang/classification/binary/40l/test-16.json')
    test_preds2 = MultiClassData.from_pred_json('assets/kangqiang/classification/binary/40l/test-28.json')
    test_gths = MultiClassData.from_gth_json('assets/kangqiang/classification/binary/40l/test-16.json')

    config = {
        'val-preds1': val_preds1,
        'val-preds2': val_preds2,
        'val-gths': val_gths,
        'test-preds1': test_preds1,
        'test-preds2': test_preds2,
        'test-gths': test_gths,
        'ParamManager': MultiClassParam,
        'Ensembler': MultiClassEnsembler,
        'Evaluator': mApEvaluator,
        'Optimizer': GeneticOptimizer,
    }
    return config

def kangqiang_classifier_2weaks():
    val_preds1 = MultiClassData.from_pred_json('assets/kangqiang/classification/multi-class/preds-wuzi-merged/val.json')
    val_preds2 = MultiClassData.from_pred_json('assets/kangqiang/classification/multi-class/preds-nonwuzi-merged/val.json')
    val_gths = MultiClassData.from_gth_json('assets/kangqiang/classification/multi-class/gths/val.json')

    test_preds1 = MultiClassData.from_pred_json('assets/kangqiang/classification/multi-class/preds-wuzi-merged/test.json')
    test_preds2 = MultiClassData.from_pred_json('assets/kangqiang/classification/multi-class/preds-nonwuzi-merged/test.json')
    test_gths = MultiClassData.from_gth_json('assets/kangqiang/classification/multi-class/gths/test.json')

    config = {
        'val-preds1': val_preds1,
        'val-preds2': val_preds2,
        'val-gths': val_gths,
        'test-preds1': test_preds1,
        'test-preds2': test_preds2,
        'test-gths': test_gths,
        'ParamManager': MultiClassParam,
        'Ensembler': MultiClassEnsembler,
        'Evaluator': mApEvaluator,
        'Optimizer': BayesianOptimizer,
    }
    return config


def kangqiang_classifier_1strong():
    """ this is used for observing the evaluation results of the strong classifications """
    val_preds1 = MultiClassData.from_pred_json('assets/kangqiang/classification/multi-class/preds/val.json')
    val_preds2 = MultiClassData.from_pred_json('assets/kangqiang/classification/multi-class/preds/val.json')
    val_gths = MultiClassData.from_gth_json('assets/kangqiang/classification/multi-class/gths/val.json')

    test_preds1 = MultiClassData.from_pred_json('assets/kangqiang/classification/multi-class/preds/test.json')
    test_preds2 = MultiClassData.from_pred_json('assets/kangqiang/classification/multi-class/preds/test.json')
    test_gths = MultiClassData.from_gth_json('assets/kangqiang/classification/multi-class/gths/test.json')

    config = {
        'val-preds1': val_preds1,
        'val-preds2': val_preds2,
        'val-gths': val_gths,
        'test-preds1': test_preds1,
        'test-preds2': test_preds2,
        'test-gths': test_gths,
        'ParamManager': MultiClassParam,
        'Ensembler': MultiClassEnsembler,
        'Evaluator': mApEvaluator,
        'Optimizer': PassiveOptimizer,
    }
    return config


def tianhe_multilabel():
    val_preds1 = MultiLabelData.from_pred_json('assets/tianhe/multi-label/preds_val1.json')
    val_preds2 = MultiLabelData.from_pred_json('assets/tianhe/multi-label/preds_val2.json')
    val_gths = MultiLabelData.from_gth_json('assets/tianhe/multi-label/preds_val1.json')

    test_preds1 = MultiLabelData.from_pred_json('assets/tianhe/multi-label/preds_test1.json')
    test_preds2 = MultiLabelData.from_pred_json('assets/tianhe/multi-label/preds_test2.json')
    test_gths = MultiLabelData.from_gth_json('assets/tianhe/multi-label/preds_test1.json')

    config = {
        'val-preds1': val_preds1,
        'val-preds2': val_preds2,
        'val-gths': val_gths,
        'test-preds1': test_preds1,
        'test-preds2': test_preds2,
        'test-gths': test_gths,
        'ParamManager': MultiLabelParam,
        'Ensembler': MultiLabelEnsembler,
        'Evaluator': PrecisionEvaluator.config(recall_thresh=.995, class_names=['cls2', 'cls3', 'cls4', 'cls5', 'cls6']),
        'Optimizer': GeneticOptimizer,
    }
    return config


def hyc_yldata():
    val_preds1 = MultiClassData.from_pred_json('assets/hyc/classification/unet++.json')
    val_preds2 = MultiClassData.from_pred_json('assets/hyc/classification/resnet.json')
    val_gths = MultiClassData.from_gth_json('assets/hyc/classification/resnet.json')

    test_preds1 = MultiClassData.from_pred_json('assets/hyc/classification/unet++.json')
    test_preds2 = MultiClassData.from_pred_json('assets/hyc/classification/resnet.json')
    test_gths = MultiClassData.from_gth_json('assets/hyc/classification/resnet.json')

    config = {
        'val-preds1': val_preds1,
        'val-preds2': val_preds2,
        'val-gths': val_gths,
        'test-preds1': test_preds1,
        'test-preds2': test_preds2,
        'test-gths': test_gths,
        'ParamManager': MultiClassParam,
        'Ensembler': MultiClassEnsembler,
        'Evaluator': PrecisionEvaluator.config(recall_thresh=.995, class_names=['NG']),
        'Optimizer': BayesianOptimizer,
    }
    return config


def canadian():
    val_preds1 = DetectionData.from_json('assets/canadian/preds1_val.json')
    val_preds2 = DetectionData.from_json('assets/canadian/preds2_val.json')
    val_gths = DetectionData.from_json('assets/canadian/labels_val.json')

    test_preds1 = DetectionData.from_json('assets/canadian/preds1_test.json')
    test_preds2 = DetectionData.from_json('assets/canadian/preds2_test.json')
    test_gths = DetectionData.from_json('assets/canadian/labels_test.json')

    config = {
        'val-preds1': val_preds1,
        'val-preds2': val_preds2,
        'val-gths': val_gths,
        'test-preds1': test_preds1,
        'test-preds2': test_preds2,
        'test-gths': test_gths,
        'ParamManager': DetectionParam,
        'Ensembler': DetectionEnsembler,
        # 'Evaluator': mApEvaluator,
        # 'Evaluator': PrecisionEvaluator.config(recall_thresh=.995, class_names=['cls1', 'cls2', 'cls3', 'cls4', 'cls5', 'cls6']),
        'Evaluator': RecallEvaluator.config(precision_thresh=.995, class_names=['cls1', 'cls2', 'cls3', 'cls4', 'cls5', 'cls6']),
        'Optimizer': BayesianOptimizer,
    }
    return config


def canadian_hybrid():
    val_preds1 = DetectionData.from_json('assets/canadian/hybrid/detection_val_pre.json')
    val_preds2 = MultiLabelData.from_pred_json('assets/canadian/hybrid/multilabel_val.json')
    val_gths = DetectionData.from_json('assets/canadian/hybrid/detection_val_gt.json')

    test_preds1 = DetectionData.from_json('assets/canadian/hybrid/detection_test_pre.json')
    test_preds2 = MultiLabelData.from_pred_json('assets/canadian/hybrid/multilabel_test.json')
    test_gths = DetectionData.from_json('assets/canadian/hybrid/detection_test_gt.json')

    config = {
        'val-preds1': val_preds1,
        'val-preds2': val_preds2,
        'val-gths': val_gths,
        'test-preds1': test_preds1,
        'test-preds2': test_preds2,
        'test-gths': test_gths,
        'ParamManager': HybridParam,
        'Ensembler': HybridEnsembler,
        'Evaluator': mApEvaluator,
        # 'Evaluator': PrecisionEvaluator.config(recall_thresh=.9, class_names=['cls2', 'cls3', 'cls4', 'cls5', 'cls6', 'cls7']),
        'Optimizer': BayesianOptimizer,
    }
    return config