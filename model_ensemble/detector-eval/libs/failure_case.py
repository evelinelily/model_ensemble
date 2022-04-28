import sys
sys.path.append('../')
from evaluator import Evaluator4Detector, Evaluator4Classifier
import cv2, os
from tqdm import tqdm
from utils import draw_instances, generate_path_dict
import pandas as pd

def my_draw_instances(img, image_instances, color):
    boxes = [inst['box'] for inst in image_instances]
    clsnames = [inst['class_name'][:2] for inst in image_instances]
    return draw_instances(img, boxes, labels=clsnames, color=color, disp_margin=0)


def draw_miss(evaluator, class_names, path_dict, save_dir, iou_threshold):
    config = {
        'iou_threshold': iou_threshold,
        'class_names': class_names,
        'class_names_pred': evaluator.class_names,
        }
    miss_dict = evaluator.find_miss_instances(config)
    if len(miss_dict) == 0:
        return   # do nothing if no miss instances found
    ## Dump missed class
    final_save_dir = os.path.join(save_dir, '+'.join(class_names))
    if not os.path.isdir(final_save_dir): os.makedirs(final_save_dir)
    for img_name, missed_instances in tqdm(miss_dict.items()):
        # get detection instances
        detection_instances = evaluator.detections[img_name]
        # load failure image
        img_file = path_dict[img_name]
        img = cv2.imread(img_file)
        # save failure image
        save_file = os.path.join(final_save_dir, img_name)
        img = my_draw_instances(img, detection_instances, [0,0,255])
        img = my_draw_instances(img, missed_instances, [0,255,0])
        cv2.imwrite(save_file, img)

    print("miss {} images and {} instances".format(len(miss_dict), \
        (sum(len(x) for x in miss_dict.values()))))


def get_miss_info_given_classnames(evaluator, class_names, iou_threshold):
    config = {
        'iou_threshold': iou_threshold,
        'class_names': class_names,
        'class_names_pred': evaluator.class_names,
        }
    miss_dict = evaluator.find_miss_instances(config)

    # dump total info
    num_instances = evaluator.annotations.get_instance_num(class_names)
    num_miss_instances = sum(len(x) for x in miss_dict.values())

    return num_miss_instances, num_instances

def record_miss_info(evaluator, iou_threshold, save_path):
    table_dict = dict()
    # set x label:
    table_dict[''] = ['#Miss / #Total', 'Miss Rate']
    # per-class info
    for class_name in evaluator.class_names:
        num_miss_instances, num_instances = get_miss_info_given_classnames(evaluator, [class_name], iou_threshold)
        table_dict[class_name] = ['{}/{}'.format(num_miss_instances, num_instances), '{:.1f}%'.format(num_miss_instances / num_instances * 100)]
    # total info
    num_miss_instances, num_instances = get_miss_info_given_classnames(evaluator, evaluator.class_names, iou_threshold)
    table_dict['Total'] = ['{}/{}'.format(num_miss_instances, num_instances), '{:.1f}%'.format(num_miss_instances / num_instances * 100)]

    df = pd.DataFrame(table_dict)
    print(df, file=open(save_path, 'w'))


def draw_all(evaluator, jm_det, path_dict, save_dir):
    """ Draw all samples with predictions and annotations on it for classification """
    assert isinstance(evaluator, Evaluator4Classifier), "Currently we only support classification evaluation"
    jm_ann = evaluator.annotations
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for img_name in tqdm(jm_ann.iterator(), total=jm_ann.image_num):
        # get detection and annotation instances
        detection_instances = jm_det[img_name]
        annotation_instances = jm_ann[img_name]
        # load failure image
        img_file = path_dict[img_name]
        img = cv2.imread(img_file)
        # save failure image
        save_file = os.path.join(save_dir, img_name)
        img = my_draw_instances(img, detection_instances,  [0,0,255])
        img = my_draw_instances(img, annotation_instances, [0,255,0])
        cv2.imwrite(save_file, img)