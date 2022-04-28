#
# This script is used to crop defect regions from ground truths (all ground truth),
# and detections (overkilled detections).
# It's part of the data clean process
#
import cv2, os
from tqdm import tqdm
import setup_env
from evaluator import Evaluator4Detector
from utils import generate_path_dict, draw_instances

def remove_extension(file_name):
    return '.'.join(file_name.split('.')[:-1])

def crop_defects(image, boxes):
    crop_margin = 80
    crops = list()
    for box in boxes:
        x1,y1,x2,y2 = box
        crops.append(
            image[max(0,y1-crop_margin):y2+crop_margin,max(0,x1-crop_margin):x2+crop_margin,:]
        )
    return crops

def save_crops(save_dir, image_name, crops):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for ix, crop in enumerate(crops):
        save_path = os.path.join(save_dir, image_name+'-crop{}.jpg'.format(ix))
        cv2.imwrite(save_path, crop)


def run(det_json, gth_json, save_dir):
    global path_dict
    ev = Evaluator4Detector(
        det_json = det_json,
        gth_json = gth_json
    )
    ev.filter_detection_results(score_threshold=.5)

    # save NG crops (from annotations)
    for image_name in tqdm(ev.annotations.iterator(), total=ev.annotations.size):
        image_path = path_dict[image_name]
        image = cv2.imread(image_path)
        boxes_dt = [inst['box'] for inst in ev.detections[image_name]]
        boxes_gt = [inst['box'] for inst in ev.annotations[image_name]]
        class_dt = [inst['class_name'] for inst in ev.detections[image_name]]
        # draw detection results and ground truths
        image = draw_instances(image, boxes_gt, color=(0,255,0), thickness=2, disp_margin=10)
        image = draw_instances(image, boxes_dt, class_dt, color=(0,0,255), thickness=1, disp_margin=10)

        boxes = [x['box'] for x in ev.annotations[image_name]]
        crops = crop_defects(image, boxes)
        save_crops(save_dir+'/NG/', remove_extension(image_name), crops)


    config = {
        'iou_threshold': .00001,
        'class_names': ev.class_names,  # all ngs
    }
    # save OK crops (from detection false alarms)
    fa_dict = ev.find_fa_instances(config)
    for image_name, contents in tqdm(fa_dict.items()):
        image_path = path_dict[image_name]
        image = cv2.imread(image_path)
        boxes_dt = [inst['box'] for inst in ev.detections[image_name]]
        boxes_gt = [inst['box'] for inst in ev.annotations[image_name]]
        class_dt = [inst['class_name'] for inst in ev.detections[image_name]]
        # draw detection results and ground truths
        image = draw_instances(image, boxes_gt, color=(0,255,0), thickness=2, disp_margin=10)
        image = draw_instances(image, boxes_dt, class_dt, color=(0,0,255), thickness=1, disp_margin=10)

        boxes = [x['box'] for x in contents]
        crops = crop_defects(image, boxes)
        save_crops(save_dir+'/OK/', remove_extension(image_name), crops)


if __name__ == '__main__':
    path_dict = generate_path_dict('/mnt/nfs/zhaimenghua/data/schaeffler/Unique_Samples/')
    run(
        gth_json = '/mnt/nfs/zhaimenghua/data/schaeffler/experiments/20200318/train-eval/data-fg/test-gt.json',
        det_json = '/mnt/nfs/zhaimenghua/data/schaeffler/experiments/20200318/infer/test/resnet50-1600-v1/det.json',
        save_dir='/home/zhai/schaeffler-crops/'
    )