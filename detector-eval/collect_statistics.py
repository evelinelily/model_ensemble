import os
import numpy as np
from collections import OrderedDict
from utils import generate_path_dict
from evaluator import Evaluator4Detector
from libs.compute_ap import get_ap

##################### xlsxwriter tutorial  ######################
# please refer to https://www.cnblogs.com/fkissx/p/5617630.html #
#################################################################
import xlsxwriter

char_table = "ABCDEFGHIJKLMNOPQRSTUVDXYZ"

def compute_stats(det_json, gt_json):
    ev = Evaluator4Detector(
        det_json = det_json,
        gth_json = gt_json
    )

    ## Compute instance level stats
    config = {
        'iou_threshold': .1,
        'class_names': ev.class_names  # all NG class
    }
    instance_res = ev.eval_instance_level(config)

    ## Find score threshold when instance recall rate = ?%
    global recall_threshold
    recall_at_all_scores = instance_res['recalls']
    all_scores = instance_res['scores']
    assert len(all_scores) == len(recall_at_all_scores)
    idx = np.abs(recall_at_all_scores - recall_threshold).argmin()
    score_threshold_at_recallx = all_scores[idx]
    print('recall: {}, score_threshold: {}'.format(recall_at_all_scores[idx], score_threshold_at_recallx))

    ## Compute mage level stats
    config = {
        'iou_threshold': .00001,        # as long as we detect one defects
        'mode': 'easy',
        'score_threshold': score_threshold_at_recallx,
        'class_names': ev.class_names,  # all NG class
    }
    image_res = ev.eval_image_level(config)

    ## compute per-class AP and mAP
    per_class_ap_dict = OrderedDict()
    for class_name in sorted(ev.class_names):
        print("compute ap for '{}'".format(class_name))
        config = {
            'iou_threshold': .1,
            'class_names': [class_name]
        }
        res = ev.eval_instance_level(config)
        per_class_ap_dict[class_name] = res['ap']

    return instance_res['ap'], per_class_ap_dict, image_res['miss'], image_res['overkill']


from collections import OrderedDict
def run():
    ## 1) collect statistics
    stat_dict = OrderedDict()
    for config in configuration_generator():
        assert os.path.exists(config['det_json']), config['det_json']
        assert os.path.exists(config['gt_json']), config['gt_json']
        ng_ap, per_class_ap_dict, miss, overkill = compute_stats(config['det_json'], config['gt_json'])
        mAP = sum(per_class_ap_dict.values()) / len(per_class_ap_dict)
        stat = {'NG_AP': ng_ap, 'mAP': mAP, 'miss': miss, 'overkill': overkill, 'per-class-ap': per_class_ap_dict}

        mode = config['mode']
        if mode not in stat_dict:
            stat_dict[mode] = OrderedDict()
        stat_dict[mode][config['exp']] = stat

    ## 2) write statistics to xlsx file
    workbook = xlsxwriter.Workbook(config['save_xlsx'])
    bold = workbook.add_format({'bold': True})  # set bold font format
    number = workbook.add_format({'num_format': '#.##', 'align': 'right'})
    number_bold = workbook.add_format({'num_format': '#.##', 'align': 'right', 'bold': True})

    # loop over modes ('test'/'val'/'test_ng_only')
    for mode in stat_dict:
        worksheet = workbook.add_worksheet(name=mode)
        worksheet.write('A1', 'Experiment', bold)
        worksheet.write('B1', 'NG AP (%)', bold)
        worksheet.write('C1', 'NG Miss (%)', bold)
        worksheet.write('D1', 'NG Overkill (%)', bold)

        # append columns for per-class aps
        assert len(per_class_ap_dict) <= 20, "Too many classes to put on the table (> 20)"
        for idx, class_name in enumerate(per_class_ap_dict):
            column_name = '{} AP'.format(class_name)
            worksheet.write('{}1'.format(char_table[idx+4]), column_name, bold)

        # append "mAP"
        worksheet.write('{}1'.format(char_table[idx+5]), 'mAP', bold)

        # loop over experiments
        for i, exp in enumerate(stat_dict[mode]):
            stat = stat_dict[mode][exp]
            worksheet.write(i+1, 0, exp)
            worksheet.write(i+1, 1, stat['NG_AP']*100, number)
            worksheet.write(i+1, 2, stat['miss']*100, number)
            worksheet.write(i+1, 3, stat['overkill']*100, number)

            # append rows for per-class aps
            per_class_ap_dict = stat['per-class-ap']
            for idx, class_ap in enumerate(per_class_ap_dict.values()):
                worksheet.write(i+1, 4+idx, class_ap*100, number)
            worksheet.write(i+1, 5+idx, stat['mAP']*100, number)


        # compute average ap across all experiments
        worksheet.write(i+2,0,'AVG', bold)
        worksheet.write(i+2,1,'=AVERAGE(B2:B{})'.format(i+1), number_bold)
        worksheet.write(i+2,2,'=AVERAGE(C2:C{})'.format(i+1), number_bold)
        worksheet.write(i+2,3,'=AVERAGE(D2:D{})'.format(i+1), number_bold)

        global recall_threshold
        worksheet.write(i+3,2,'when recall={:.2f}'.format(recall_threshold), bold)

        # set column width
        worksheet.set_column(0,0,14)
        worksheet.set_column(0,1, 5)
        worksheet.set_column(0,2,14)
        worksheet.set_column(0,3,14)

    workbook.close()



from experiments import experiment_kangqiang_24l as configuration_generator
recall_threshold = .9   # instance recall threshold
if __name__ == '__main__':
    run()