import os
from utils import generate_path_dict
from evaluator import Evaluator4Detector
from libs.compute_ap import draw_pr_curves
from libs.failure_case import draw_miss, record_miss_info

def run_evaluation(det_json, gt_json, save_dir):
    iou_threshold_detection=.1
    global path_dict

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ev = Evaluator4Detector(
        det_json = det_json,
        gth_json = gt_json
    )
    #
    # Draw confusion matrix
    #
    ev.draw_confusion_matrix(iou_threshold_detection, save_path=os.path.join(save_dir, 'cm.png'))

    # #
    # # AI detection: draw detection failure cases (miss)
    # #
    # miss_dir = os.path.join(save_dir, 'miss/')
    # for class_name in ev.label2class.values():
    #     print(class_name)
    #     draw_miss(ev, [class_name], path_dict, miss_dir, iou_threshold_detection)
    
    miss_info_txt = os.path.join(save_dir, 'miss_info.txt')
    record_miss_info(ev, iou_threshold_detection, miss_info_txt)


    #
    # Draw PR curves for AI detection
    #
    draw_pr_curves(ev, iou_threshold_detection, save_dir=os.path.join(save_dir, 'pr_curves/'))



from experiments import experiment_kangqiang_24l as configuration_generator
if __name__ == '__main__':
    for config in configuration_generator():
        assert os.path.exists(config['det_json']), config['det_json']
        assert os.path.exists(config['gt_json']), config['gt_json']
        assert os.path.isdir(config['img_dir']), config['img_dir']
        path_dict = generate_path_dict(config['img_dir'])  # path_dict is a global variable
        run_evaluation(config['det_json'], config['gt_json'], config['save_dir'])