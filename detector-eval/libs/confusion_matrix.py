import sys
sys.path.append('..')

import cv2, os
from evaluator import Evaluator4Classifier as Evaluator
from utils import draw_instances, generate_path_dict
from tqdm import tqdm
import pandas as pd

def get_image_instances_in_cm(evaluator, pred_name, ann_name):
    """
    Get images and their prediction and annotation instances
    param:
        evaluator: instance of the Evaluator class
        pred_name: class names of predictions
        ann_name: class names of annotations
    return:
        pred_instances: a dictionary that indexed by image names, it records the bbox info of predictions
        ann_instances: a dictionary that indexed by image names, it records the bbox info of annotations
    """
    pred_instances = dict()
    ann_instances = dict()
    for img_name, (contents_det, contents_ann) in evaluator.items():
        tmp_pred, tmp_ann = list(), list()
        for cont_det, cont_ann in zip(contents_det, contents_ann):
            if cont_det['class_name'] == pred_name and \
                cont_ann['class_name'] == ann_name:
                tmp_pred.append(cont_det)
                tmp_ann.append(cont_ann)
        if len(tmp_pred) > 0:
            pred_instances[img_name] = tmp_pred
            ann_instances[img_name] = tmp_ann
    return pred_instances, ann_instances


def draw_samples_in_cm(evaluator, det_instances, path_dict, save_path):
    for pred_name in evaluator.class_names:
        for ann_name in evaluator.class_names:
            if pred_name == ann_name:
                continue
            else:
                # get mis-classified instances
                pred_instances, ann_instances = \
                    get_image_instances_in_cm(evaluator, pred_name, ann_name)

                # skip correct pred/anno class pairs
                if len(pred_instances) == 0:
                    continue

                print(pred_name, ann_name)

                # loop over the found images containing mis-classifed instances
                save_folder = os.path.join(save_path, ann_name+'+'+pred_name)
                if not os.path.exists(save_folder): os.makedirs(save_folder)
                for img_name in tqdm(pred_instances):
                    # prepare instances to draw on the image
                    # boxes_pred = [x['box'] for x in pred_instances[img_name]]
                    # labels_pred = [x['class_name'][:2] for x in pred_instances[img_name]]
                    boxes_ann = [x['box'] for x in ann_instances[img_name]]
                    labels_ann = [x['class_name'][:2] for x in ann_instances[img_name]]

                    # extract all detection results
                    boxes_det = [x['box'] for x in det_instances[img_name]]
                    labels_det = [x['class_name'][:2] for x in det_instances[img_name]]

                    # load image from disk
                    image = cv2.imread(path_dict[img_name])

                    # draw instances on the image
                    # image = draw_instances(image, boxes_pred, labels_pred, color=(0,0,255))
                    image = draw_instances(image, boxes_ann, labels_ann, color=(0,255,0))
                    image = draw_instances(image, boxes_det, labels_det, color=(0,0,255))

                    # save the image to the given path
                    save_file = os.path.join(save_folder, img_name)
                    cv2.imwrite(save_file, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def record_precision_info(evaluator, save_path):
    """
    Get instance level classification precision for structural defect classes (lianjiao, shakong etc.)
    param:
        evaluator: instance of the Evaluator class
        save_path: the txt file to which you want to save the precision info
    """
    # split the save_path into two sub save_path
    basename = save_path.rstrip('t').rstrip('x').rstrip('t').rstrip('.')  # remove the extension .txt

    # regard lianjiao + shakong as structual defects
    save_path_lish = basename + '-lish.txt'
    structural_classnames = ['shakong', 'guoshi', 'lianjiao', 'pifeng', 'lish']  # lish = lianjiao + shakong
    record_precision_info_func(evaluator, save_path_lish, structural_classnames)

    # regard only lianjiao (sometimes lish) as structual defects
    save_path_lian = basename + '-lian.txt'
    structural_classnames = ['lianjiao', 'pifeng', 'lish']  # lish = lianjiao + shakong
    record_precision_info_func(evaluator, save_path_lian, structural_classnames)



def record_precision_info_func(evaluator, save_path, structural_classnames):
    """
    Get instance level classification precision for structural defect classes (lianjiao, shakong etc.)
    param:
        evaluator: instance of the Evaluator class
        save_path: the txt file to which you want to save the precision info
        structual_classnames: definition of all POSSIBLE structual classes (or say hazardous classes)
    """
    ## print structual_classnames mis-rate and false alarm rate
    num_predicted_instances = evaluator.detections.get_instance_num(structural_classnames)
    num_annotated_instances = evaluator.annotations.get_instance_num(structural_classnames)
    num_misclassification = 0
    num_misses = 0
    for _, (contents_det, contents_ann) in evaluator.items():
        for cont_det, cont_ann in zip(contents_det, contents_ann):
            if cont_det['class_name'] in structural_classnames and \
                cont_ann['class_name'] not in structural_classnames:
                num_misclassification += 1
            if cont_ann['class_name'] in structural_classnames and \
                cont_det['class_name'] not in structural_classnames:
                num_misses += 1
    misclassification_rate = num_misclassification / num_predicted_instances
    miss_rate = num_misses / num_annotated_instances
    print(
        "Structural Defects Mis-classification Rate: #(FP) / #(TP+FP) = {} / {} = {:.1f}%".format(
            num_misclassification, num_predicted_instances, misclassification_rate*100),
            file=open(save_path, 'w')
        )
    print(
        "Structural Defects Miss Rate: #FN / #(TP+FN) = {} / {} = {:.1f}%".format(
            num_misses, num_annotated_instances, miss_rate*100),
            file=open(save_path, 'a')
        )

    ## print misclassification per class
    table_dict = dict()
    table_dict[''] = ['FP/(TP+FP)', 'Miscls Rate']
    for class_name in evaluator.class_names:
        if class_name in structural_classnames:  # skip structural class names
            continue
        num_misclassification = 0
        for _, (contents_det, contents_ann) in evaluator.items():
            for cont_det, cont_ann in zip(contents_det, contents_ann):
                if cont_det['class_name'] in structural_classnames and \
                    cont_ann['class_name'] not in structural_classnames and cont_ann['class_name'] == class_name:
                    num_misclassification += 1
        misclassification_rate = num_misclassification / num_predicted_instances
        table_dict[class_name] = [
            '{}/{}'.format(num_misclassification, num_predicted_instances),
            '{:.1f}'.format(misclassification_rate*100)
        ]
    df = pd.DataFrame(table_dict)
    print(df, file=open(save_path, 'a'))

    ## print total accuracy
    num_hit = 0
    num_instances = 0
    for _, (contents_det, contents_ann) in evaluator.items():
        for cont_det, cont_ann in zip(contents_det, contents_ann):
            if cont_det['class_name'] == cont_ann['class_name']:
                num_hit += 1
            num_instances += 1
    total_accuracy = num_hit / num_instances
    print(
        'Total Accuracy (#hits / #instances): {} / {} = {:.1f}%'.format(num_hit, num_instances, total_accuracy*100),
        file=open(save_path, 'a'))