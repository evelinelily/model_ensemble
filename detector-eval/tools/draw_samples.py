#
# Draw and save all failure cases (miss and false-alarm)
#
import os, cv2
from tqdm import tqdm
import setup_env
import numpy as np
from utils import generate_path_dict, draw_instances
from evaluator import Evaluator4Detector

def draw_mode(det_json, gt_json, save_dir, config):
    assert config['mode'] in ['failure', 'success', 'all', 'miss', 'overkill']
    global path_dict

    ev = Evaluator4Detector(
        det_json = det_json,
        gth_json = gt_json
    )

    ## Find score threshold when instance recall rate = ?%
    if 'recall_threshold' in config:
        config_tmp = {
            'iou_threshold': .1,
            'class_names': ev.class_names  # all NG class
        }
        instance_res = ev.eval_instance_level(config_tmp)
        recall_at_all_scores = instance_res['recalls']
        all_scores = instance_res['scores']
        assert len(all_scores) == len(recall_at_all_scores)
        idx = np.abs(recall_at_all_scores - config['recall_threshold']).argmin()
        score_threshold_at_recallx = all_scores[idx]
        print('recall: {}, score_threshold: {}'.format(recall_at_all_scores[idx], score_threshold_at_recallx))
        score_threshold = score_threshold_at_recallx
    elif 'score_threshold' in config:
        score_threshold = config['score_threshold']
    else:
        assert 'score_threshold' in config or 'recall_threshold' in config

    ev.filter_detection_results(score_threshold=score_threshold)

    ev_config = {
        'iou_threshold': config['iou_threshold'],
        'class_names': ev.class_names,  # all ngs
    }
    # find failure cases: misses + false alarms
    miss_dict = ev.find_miss_instances(ev_config)
    fa_dict = ev.find_fa_instances(ev_config)

    if None: pass
    elif config['mode'] is 'success':
        all_image_names = set(ev.annotations.image_dict.keys())
        selected_image_names = set(miss_dict.keys()).union(set(fa_dict.keys()))
        assert selected_image_names.issubset(all_image_names)
        selected_image_names = all_image_names.difference(selected_image_names)
    elif config['mode'] is 'all':
        selected_image_names = set(ev.annotations.image_dict.keys())
    elif config['mode'] is 'failure':
        selected_image_names = set(miss_dict.keys()).union(set(fa_dict.keys()))
    elif config['mode'] is 'miss':
        selected_image_names = set(miss_dict.keys())
    elif config['mode'] is 'overkill':
        selected_image_names = set(fa_dict.keys())
    else:
        pass

    for img_name in tqdm(selected_image_names):
        # get detection instances
        boxes_dt = [inst['box'] for inst in ev.detections[img_name]]
        boxes_gt = [inst['box'] for inst in ev.annotations[img_name]]
        # load failure image
        img_file = path_dict[img_name]
        img = cv2.imread(img_file)
        # draw detection results and ground truths
        img_show = draw_instances(img, boxes_gt, color=(0,255,0), thickness=2, disp_margin=10)
        img_show = draw_instances(img_show, boxes_dt, color=(0,0,255), thickness=1, disp_margin=10)

        ## Dump missed class
        save_file = os.path.join(save_dir, img_name)
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        cv2.imwrite(save_file, img_show)

    print("found {} / {} {} images".format(
        len(selected_image_names),
        config['mode'],
        ev.annotations.size)
    )

from experiments import experiment_kangqiang_24l as configuration_generator
if __name__ == '__main__':
    for config in configuration_generator():
        assert os.path.exists(config['det_json']), config['det_json']
        assert os.path.exists(config['gt_json']), config['gt_json']
        assert os.path.isdir(config['img_dir']), config['img_dir']

        path_dict = generate_path_dict(config['img_dir'], 'bmp')  # path_dict is a global variable
        for mode in ['miss', 'overkill']:
            ev_config = {
                'mode': mode,
                'iou_threshold': .1,
                'recall_threshold': .95
            }
            draw_mode(config['det_json'], config['gt_json'], config['save_dir']+'/failures/'+mode, config=ev_config)