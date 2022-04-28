import os
import time
import copy
import json
import pickle
import numpy as np


def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as fid:
        return pickle.load(fid)


def dump_pickle(res, pickle_file):
    with open(pickle_file, 'wb') as fid:
        return pickle.dump(res, fid)


def dump_json(res, json_file):
    with open(json_file, 'w') as fid:
        return json.dump(res, fid)


def add_ok_prob(preds):
    preds_out = []
    for image_pred in preds:
        image_pred_out = []
        for inst_pred in image_pred:
            fg_dists = copy.deepcopy(inst_pred[0])
            ok_prob = 1. - sum(fg_dists)
            box = copy.deepcopy(inst_pred[1])
            # if ok_prob >= 0: print(sum(fg_dists))
            ok_prob = max(ok_prob, 0)
            image_pred_out.append([[ok_prob] + fg_dists, box])
        preds_out.append(image_pred_out)
    return preds_out


def convert_format_lcz(res):
    """Convert the data format from me_api to lcz.
    """

    res_new = []
    for res_per_img in res:
        res_per_img_new = []
        if res_per_img:
            for res_per_inst in res_per_img:
                score_list = res_per_inst[0]
                bbox = res_per_inst[1]
                score_list.pop(0)  # Del the OK score.
                score = max(score_list)
                label = score_list.index(score)
                res_per_inst_new = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), score, label]
                res_per_img_new.append(res_per_inst_new)
        res_new.append(res_per_img_new)
    return res_new


def convert_normal_format(res):
    """Convert the data format from me_api to normal input.
    """
    label_name_list = [0, 1, 2, 3, 8, 10]
    res_new = []
    for res_per_img in res:
        res_per_img_new = []
        if res_per_img:
            for res_per_inst in res_per_img:
                score_list = res_per_inst[0]
                bbox = res_per_inst[1]
                score_list.pop(0)  # Del the OK score.
                score = max(score_list)
                label = label_name_list[score_list.index(score)]
                res_per_inst_new = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), score, label]
                res_per_img_new.append(res_per_inst_new)
        res_new.append(res_per_img_new)
    return res_new

def convert_format_me(res, class_id_list):
    """将检测结果格式转换为me-api的格式。注意，由于EL、VI检测的类别不同，此转换函数也有所不同。
    """
    ng_cls_cnt = len(class_id_list)
    res_new = []
    for res_per_img in res:
        res_per_img_new = []
        for res_per_inst in res_per_img:
            bbox = res_per_inst[0:4]
            score = res_per_inst[4]
            label = res_per_inst[5]
            score_avg = 0
            score_list = [score_avg for _ in range(ng_cls_cnt)]
            score_list[class_id_list.index(int(label))] = score
            res_per_inst_new = (score_list, bbox)
            res_per_img_new.append(res_per_inst_new)
        res_new.append(res_per_img_new)
    # Add prob for OK class.
    res_new = add_ok_prob(res_new)
    return res_new

def filter_out_result(result, score=0.):

    """过滤分数小于score的框。

    Args：
        result: y_pred。
        score：过滤分数，即分数小鱼score的框都丢掉。
    Return:
        过滤后的y_pred。
    """

    cnt1, cnt2 = 0, 0
    result_new = []
    for res in result:
        if res:
            inst_list = []
            for inst in res:
                cnt1 += 1
                if inst[4] >= score:
                    cnt2 += 1
                    inst_list.append(inst)
            result_new.append(inst_list)
        else:
            result_new.append(res)
    print('过滤前和过滤后框的总数分别为：{}, {}。'.format(cnt1, cnt2))
    return result_new


if __name__ == '__main__':
    pass