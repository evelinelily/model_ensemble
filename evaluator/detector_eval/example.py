import os
import pickle
import numpy as np
from eval_metrics import DetectorEval, get_best_thresh
from filter_bbox import filter_0, filter_2, filter_3, filter_10


def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as fid:
        return pickle.load(fid)

def dump_pickle(res, pickle_file):
    with open(pickle_file, 'wb') as fid:
        return pickle.dump(res, fid)

def get_ok_ng_img_num(gts):
    ok_num, ng_num = 0, 0
    for gt in gts:
        if gt:
            ng_num += 1
        else:
            ok_num += 1
    return ok_num, ng_num

def get_inst_num(gts, ng_cls_cnt):
    inst_cnt_list = [0 for _ in range(ng_cls_cnt)]
    for gt_per_img in gts:
        for gt_per_inst in gt_per_img:
            inst_cnt_list[gt_per_inst[5]] += 1
    return inst_cnt_list

def filter_result(gts, preds, paths, cls_id=0):
    """获取指定类别的结果，并将无关类别去除
    """
    gts_new, preds_new, paths_new = [], [], []
    cnt_ng, cnt_ok = 0, 0
    for gt_per_img, pred_per_img, path_per_img in zip(gts, preds, paths):
        if gt_per_img:
            for gt_per_inst in gt_per_img:
                # 只收集一次
                if gt_per_inst[5] == cls_id:
                    gts_new.append(gt_per_img)
                    preds_new.append(pred_per_img)
                    paths_new.append(path_per_img)
                    cnt_ng += 1
                    break
        else:
            gts_new.append(gt_per_img)
            preds_new.append(pred_per_img)
            paths_new.append(path_per_img)
            cnt_ok += 1
    print('NG image num of class {} is: {}, OK image num is: {}.'.format(cls_id, cnt_ng, cnt_ok))
    # 去除无关类别
    gts_new_refined, preds_new_refined = [], []
    for gt_per_img, pred_per_img in zip(gts_new, preds_new):
        gt_per_img_new = []
        pred_per_img_new = []
        if gt_per_img:
            for gt_per_inst in gt_per_img:
                if gt_per_inst[5] == cls_id:
                    gt_per_img_new.append(gt_per_inst)
        gts_new_refined.append(gt_per_img_new)

        if pred_per_img:
            for pred_per_inst in pred_per_img:
                # if pred_per_inst[5] == cls_id: # 只收集该类别
                if pred_per_inst[5] == cls_id:
                    pred_per_img_new.append(pred_per_inst)
        preds_new_refined.append(pred_per_img_new)

    return gts_new_refined, preds_new_refined, paths_new

def get_min_pred_score(preds):
    min_score = 1
    for pred in preds:
        if pred:
            for p in pred:
                if p[4] < min_score:
                    min_score = p[4]
    return min_score


if __name__ == '__main__':

    ################### Step 1. 获取基本的模型和数据信息 ###################
    dataset = 'models_0819_poly'
    split = 'val' # 'val' or 'test'
    model = 'fused'
    data_dir = os.path.join('./input', dataset, split)
    pred_path = os.path.join(data_dir, '{}.pkl'.format(model))
    gt_path = os.path.join(data_dir, 'gt.pkl')
    img_path = os.path.join(data_dir, 'img_path.pkl')
    pred = load_pickle(pred_path)
    gt = load_pickle(gt_path)
    paths = load_pickle(img_path)

    # # # 过滤掉不合格的框
    pred = filter_0(pred, define_corner=0.025, tolerated_ratio=0.05, cut_box=[295, 590], score_th=0.5, common_score_th=0.2844)

    pred = filter_2(pred, define_boundary=0.05, tolerated_len_ratio=0.1, common_score_th=0.2844, cut_box=[295, 590])

    pred = filter_3(pred, define_boundary=0.05, tolerated_ratio=0.2, common_score_th=0.2844, cut_box=[295, 590])

    pred = filter_10(pred, tolerated_area_ratio=0.025, cut_box=[295, 590])

    # 统计图片数目
    ok_num, ng_num = get_ok_ng_img_num(gt)
    print('NG image num is: {}, OK image num is: {}.'.format(ng_num, ok_num))

    ## 修改图片路径
    # if split == 'test':
    #     parent_path = '/home/changzhiluo/nfs-75/project_data/atesi/recognition/poly/data_processed/1_1/'
    # elif split == 'val':
    #     parent_path = '/home/changzhiluo/nfs-75/project_data/atesi/recognition/poly/data_processed/1_1/'
    # paths = [os.path.join(parent_path, p + '.jpg') for p in paths]
    # paths = [p.replace('/disk1/tmp', '/home/changzhiluo/nfs-tmp-75') for p in paths]
    paths = [p.replace('/disk', '/home/changzhiluo/nfs-75') for p in paths]
    ###################################################################

    ########## Step 2. 在val集合上获取最优决策阈值（各类别一致） ############
    ng_cls_num = 6
    iou_thresh = 0.1

    # if split == 'val':
    #     min_score = get_best_thresh(gt, pred, fnr_thresh=0.005)
    #     score_thresh = [np.round(min_score, 4) for _ in range(ng_cls_num)]
    # elif split == 'test':
    #     score_thresh = [0.2844, 0.2844, 0.2844, 0.2844, 0.2844, 0.2844]

    # 自定义阈值
    score_thresh = [0.2844, 0.2844, 0.2844, 0.2844, 0.2844, 0.2844]
    print('score_thresh is: ', score_thresh)
    ##################################################################

    #################### Step 3. 在测试集上评估结果 #####################
    """ 评估多类结果 """

    # Step 1. 初始化检测类，binary_mode设为False
    det_eval = DetectorEval(gt, pred, binary_mode=False, iou=iou_thresh, score=score_thresh)

    # Step 2. 计算结果
    ap_res = det_eval.compute_ap()
    p_at_r = det_eval.compute_p_at_r()
    print('AP res is: ', ap_res)
    # print('P at R res is: ', p_at_r)
    # det_eval.draw_pr_curve(output_dir='pr_curve')

    """ 评估两类结果（所有缺陷类别合并为1类） """

    # Step 1. 初始化检测类，binary_mode设为True
    det_eval = DetectorEval(gt, pred, binary_mode=True, iou=iou_thresh, score=score_thresh)

    # Step 2. 计算结果
    ap_res = det_eval.compute_ap()
    print('Binary AP res is: ', ap_res)
    lw_res, fn_ind_list, fp_ind_list= det_eval.compute_fnr_and_fpr(fail_study=True)
    print('loubao is {}, wubao is {}.'.format(lw_res['fnr'], lw_res['fpr']))
    det_eval.draw_pr_curve(output_dir=os.path.join('pr_curve', dataset, split, model))
    statistics = det_eval.get_failure_case_statistics(fn_ind_list, fp_ind_list, label_map={0:'0', 1:'1', 2:'2', 3:'3', 4:'8', 5:'10'})
    print('falure case statistics is: ')
    for stat in statistics:
        print(stat)
    det_eval.draw_failure_cases(paths, fn_ind_list, fp_ind_list, label_map={0:'0', 1:'1', 2:'2', 3:'3', 4:'8', 5:'10'}, show_scores=True, res_dir=os.path.join('failure_cases', dataset, split, model))
