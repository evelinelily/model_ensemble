#
# A demonstration of model ensemble (ME), including:
# 1) ME hyper-parameter optimization
# 2) application of the model ensemble on single images
#
import sys, pickle
sys.path.append('model_ensemble/')

# import necessary classes for detection model ensemble
# all possible mode: ["detection", "multi-class", "multi-label"]
from me_api import import_me_classes
ModelEnsemble, ParameterClass = import_me_classes(mode='hybrid')

# Prepare data
with open('assets/data.pkl', 'rb') as fid:
    dat = pickle.load(fid)
predictions1_val = dat['val_preds1']
predictions2_val = dat['val_preds2-multilabel']
ground_truth_val = dat['val_gths']
predictions1_test = dat['test_preds1']
predictions2_test = dat['test_preds2-multilabel']


if __name__ == '__main__':
    ## 1) optimize hyper-parameters on validation data
    me_for_optimization = ModelEnsemble()
    # feed validation data to the optimizer
    me_for_optimization.initialize_optimizer(
        predictions1 = predictions1_val,
        predictions2 = predictions2_val,
        ground_truth = ground_truth_val
    )
    label2class = me_for_optimization.fake_class_dict(ground_truth_val)
    print("Dataset: {} classes".format(len(label2class)))
    # # set metric to optimize, and the optimizer type
    # optimal_param = me_for_optimization.maximize(  # A complicated metric optimization
    #     metric='rec@prec', optimizer='bayesian',
    #     class_names=['class5', 'class2'], iou_threshold=0.1, precision_threshold=.4
    # )
    optimal_param = me_for_optimization.maximize(metric='mAP', optimizer='passive')
    optimal_param.dump()                                   # display the optimal hyper-parameters on terminal
    optimal_param.pickle(pickle_path='optim_param.pkl')    # save the hyper-parameters on disk (to be loaded later)

    ## 2) apply model ensemble
    me = ModelEnsemble()
    me.initialize_parameter(
        parameter=ParameterClass.from_pickle(pickle_path='optim_param.pkl')
    )
    for pred1, pred2 in zip(predictions1_test, predictions2_test):
        pred_fused = me.fuse(pred1, pred2)
        print(pred_fused)