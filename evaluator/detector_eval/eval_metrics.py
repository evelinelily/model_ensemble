
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from collections import defaultdict
from time import time
from sklearn import metrics


def get_best_thresh(y_true, y_pred, fnr_thresh=0.005):
    """在验证集上获取目标检测最优决策阈值（各类别一致）。

    注意：此处设定的fnr_thresh可能会和最终评估出来的fnr有一定差异，因为这里是按multilabel的标准选的score_thresh，没有考虑iou，但实际在评估目标检测时会考虑iou。

    Args:
        y_true: 验证集gt
        y_pred: 验证集pred
    Returns:
        min_score: 能得到小于等于fnr_thresh的漏报的最小score_thresh，为一个标量。
    """
    # 1. 将检测结果转为二分类结果
    labels = []
    predicts = []
    for per_img_true in y_true:
        if per_img_true:
            labels.append(1)
        else:
            labels.append(0)
    for per_img_pred in y_pred:
        if per_img_pred:
            max_score = 0
            for per_inst_pred in per_img_pred:
                if per_inst_pred[4] > max_score:
                    max_score = per_inst_pred[4]
            predicts.append(max_score)
        else:
            predicts.append(0)
    # 2. 按照二分类标准获取最优阈值
    _, recall, thresholds = metrics.precision_recall_curve(labels, predicts)
    ind_all = np.where(recall >= (1 - fnr_thresh))
    if not list(ind_all[0]): 
        print('没有符合要求的漏报率，取最低漏报率对应的阈值！')
        min_score = thresholds[0]
    else:
        ind = ind_all[0][-1]
        min_score = thresholds[ind]
    return min_score

def convert_det_to_multilabel(y_pred, y_true):
    """将检测结果转为multilabel分类的结果。主要用于显示一张图上各个类别的score。适用于y_pred。y_true用于获取ng类别数目。
    """
    # 1. 获取ng类别数目
    ng_cls_list = []
    for per_img_true in y_true:
        if per_img_true:
            for per_inst_true in per_img_true:
                lab = per_inst_true[5]
                if lab not in ng_cls_list:
                    ng_cls_list.append(lab)
    ng_cls_list.sort()
    # print('All NG class_ids are: ', ng_cls_list)
    ng_cls_num = len(ng_cls_list)

    # 2. 转换y_pred
    y_pred_new = []
    for per_img_pred in y_pred:
        per_img_pred_new = [0 for i in range(ng_cls_num)]
        if per_img_pred:
            for per_inst_pred in per_img_pred:
                lab = int(per_inst_pred[5])
                score = np.round(per_inst_pred[4], 4)
                if score > per_img_pred_new[lab]:
                    per_img_pred_new[lab] = score 
        y_pred_new.append(per_img_pred_new)
    return y_pred_new

class DetectorEval(object):
    """一个用于评价目标检测模型的类，优点：
        1） 含AP，P@R，漏报+误报等缺陷检测中常用的指标。
        2） 使用方法简单。
        3） 评估速度快。计算2000张图片的ap，总耗时不到0.3秒，见示例。
    注意：
        此处AP的计算和学术界目标检测的AP计算方法稍微有点不同，学术界目标检测的AP计算是不考虑OK样本的，但是我们这里考虑了。
    """

    def __init__(self, y_true, y_pred, binary_mode=False, iou=0.5, score=[]):
        """构造函数，获取评价需要的参数。

        Args:
            y_pred:  
                检测模型产生的结果, 是一个3层的list。第1层是图片级别信息的集合；第2层是bbox级别信息的集合；第3层是单个bbox的结果，其格式为[x_min, y_min, x_max, y_max, score, class_id]。注意class_id必须从0开始，按顺序拍下去，即0, 1, 2, ...。 示例：[[], [[493, 56, 539, 78, 0.9686, 1], [200, 222, 217, 263, 0.9219, 1]], [], [[526, 336, 542, 378, 0.7786, 1]]]。
            y_true:  
                ground truth值，格式和y_pred完全一致，只是其score不会被用到，可以随便填一个值。
            binary_mode: 
                True时将各个类别合并为1类，否则不合并。
            iou:     
                iou阈值。检测框和真实框的交并比大于等于iou时才算检测到，反之没检测到。默认值为0.5。
            score：  
                score阈值。做评价之前可以先过滤自信度小于score的框。默认score为0，即不过滤。
        """
        assert(len(y_true) == len(y_pred))
        assert(iou >= 0)
        # 获取用于显示的score_list, 注意须在filter_out_result前做。
        self.y_pred_multilabel = convert_det_to_multilabel(y_pred, y_true)
        self.iou = iou - 1e-10
        self.score = score
        self.y_true = y_true
        self.y_pred = self.filter_out_result(y_pred)
        self.binary_mode = binary_mode
        self.gts, self.dts = self.reorganize_result()
        self.class_ids = self.get_class_ids()
        self.image_ids = self.get_image_ids()
        self.gt_inst_num = self.get_gt_inst_num()
        self.pr_dict = self.compute_precision_and_recall()

    def filter_out_result(self, result):
        """过滤score小于self.score的结果, 作用于y_pred。注意self.score为一个list， 即各个类别的过滤分数可能不一样。

        Args：
            result: y_pred。
        Return:
            过滤后的y_pred。
        """
        if not self.score:
            print('无过滤条件，不对结果进行过滤。')
            return result
        result_new = []
        for res in result:
            if res:
                inst_list = []
                for inst in res:
                    if inst[4] >= self.score[int(inst[5])]:
                        inst_list.append(inst)
                result_new.append(inst_list)
            else:
                result_new.append(res)
        return result_new

    def compute_ap(self):
        """计算每个类别的AP，计算方法：AP = sum((R_(n) - R_(n-1)) * P_(n))。

        Args:
            无
        Return：
            各个类别的class_id和AP一一对应的结果。
        """
        ap_res = dict()
        for pr_item in self.pr_dict:
            # 获取当前类别的precision和recall
            precision = pr_item['precision']
            recall = pr_item['recall']
            cls_id = pr_item['class_id']
            # 计算当前类别的AP
            sum_ap = 0
            for i in range(0, recall.shape[0] - 1):
                sum_ap += (recall[i + 1] - recall[i]) * precision[i + 1]
            ap_res[cls_id] = round(sum_ap, 4)
        return ap_res

    def compute_p_at_r(self, recall_thresh=0.995):
        """计算每个类别的precision@recall。如果recall到达不了给定阈值，则返回真实的最大recall（相应的precision不为0）对应的precision。

        Args：
            recall_thresh： recall的阈值
        Return：
            各个类别的precision@(recall >= recall_thresh)
        """
        assert(recall_thresh >= 0 and recall_thresh <= 1)
        p_at_r_list = []
        for pr_item in self.pr_dict:
            # 获取当前类别的precision和recall，注意recall和precision分别按从小到大和从大到小排好序了。
            res = dict()
            precision = pr_item['precision']
            recall = pr_item['recall']
            cls_id = pr_item['class_id']
            ind_recall = np.where(recall >= recall_thresh)
            # 由于计算precision和recall时将头尾都加上了，因此这里一定能取到recall大于0.995的情况。
            assert(ind_recall[0].shape[0] != 0)
            ind_r = ind_recall[0][0]
            if precision[ind_r] != 0:
                res['recall'] = round(recall_thresh, 4)
                res['precision'] = round(precision[ind_r], 4)
                res['class_id'] = cls_id
            else:
                ind_precision = np.where(precision > 0)
                ind_p = ind_precision[0][-1]
                res['recall'] = round(recall[ind_p], 4)
                res['precision'] = round(precision[ind_p], 4)
                res['class_id'] = cls_id
            p_at_r_list.append(res)
        return p_at_r_list

    def draw_pr_curve(self, output_dir='pr_curve'):
        """分别画每个类别的pr曲线。

        Args:
            y_true:  1维nparray，所有样本的真值列表[nSamples], 以0和1形式给出， e.g., [0, 0, 1, 0]
            y_score: 1维nparray，所有样本在正类上的分数, e.g., [0.2, 0.9, 0.7, 0.1]
        Return:
            无
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for pr_item in self.pr_dict:
            # 获取当前类别的precision和recall
            precision = pr_item['precision']
            recall = pr_item['recall']
            cls_id = pr_item['class_id']
            # 画当前类别的pr曲线
            output_path = os.path.join(output_dir, 'pr_curve_for_cls_' + str(cls_id) + '.jpg')
            plt.figure()
            plt.step(recall, precision, where='pre')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([-0.05, 1.05])
            plt.xlim([-0.05, 1.05])
            plt.title('PR Curve of Class ' + str(cls_id))
            plt.savefig(output_path)
            plt.close()

    def compute_fnr_and_fpr(self, fail_study=False):
        """计算漏报率和误报率。具体计算参考：https://en.wikipedia.org/wiki/Receiver_operating_characteristic。
            fnr = 1 - tpr = fn / (tp + fn), miss rate （漏报率）
            fpr = fp / (tn + fp), false alarm rate （误报率）
            注意： self.iou和self.score对漏报和误报的影响较大，可以通过调节这两个参数来达到期望的漏报率和误报率。具体来说：self.score会影响漏报和误报，self.iou只会影响漏报。

        Args:
            fail_study: True时返回失败案例的索引，含漏报和误报的，False时返回两个空list。
        Return：
            res, 含fnr和fpr。
            fn_ind_list: fail_study为False时为[]，否则为漏报的图片的index list
            fp_ind_list: fail_study为False时为[]，否则为误报的图片的index list
            其中，fn_ind_list、fp_ind_list用于draw_failure_case
        """
        fn_ind_list, fp_ind_list = [], []
        dt_res_all = self.collect_results_per_image()
        tp, fp, tn, fn = 0, 0, 0, 0
        if fail_study:
            for i, y_true_per_img in enumerate(self.y_true):
                dt_res_per_img = dt_res_all[i]
                if y_true_per_img:
                    if 1 in dt_res_per_img:
                        tp += 1
                    else:
                        fn += 1
                        fn_ind_list.append(i)
                elif not y_true_per_img:
                    if dt_res_per_img:
                        fp += 1
                        fp_ind_list.append(i)
                    else:
                        tn += 1
        else:
            for i, y_true_per_img in enumerate(self.y_true):
                dt_res_per_img = dt_res_all[i]
                if y_true_per_img:
                    if 1 in dt_res_per_img:
                        tp += 1
                    else:
                        fn += 1
                elif not y_true_per_img:
                    if dt_res_per_img:
                        fp += 1
                    else:
                        tn += 1
        fnr = fn / (tp + fn)
        fpr = fp / (tn + fp)
        res = dict()
        res['fnr'] = round(fnr, 4)
        res['fpr'] = round(fpr, 4)
        return res, fn_ind_list, fp_ind_list

    def get_failure_case_statistics(self, fn_index_list, fp_index_list, label_map):
        """获取各个类别的failure case的统计信息，含漏报和误报的。漏报只给出数目，误报除了给出数目外，还给出分数统计信息（最小分值, 中位数分值, 最大分值）。

        Args:
            fn_index_list: 漏报的样本索引
            fp_index_list: 误报的样本索引
            label_map: 一个映射字典, 如{0:'0', 1:'1', 2:'2', 3:'3', 4:'8', 5:'10'}
            res_dir: failure case存放路径
        Return:
            res_all: 所有缺陷统计信息
        """
        
        fnr_list = [0 for i in range(len(label_map))]
        if not fn_index_list:
            print('fn_index_list is empty!')
        else:
            for ind in fn_index_list:
                gt_label = self.y_true[ind]
                for lab in gt_label:
                    fnr_list[int(lab[5])] += 1

        fpr_list = [0 for i in range(len(label_map))]
        fpr_score_list = [[] for i in range(len(label_map))]
        if not fp_index_list:
            print('fp_index_list is empty!')
        else:
            for ind in fp_index_list:
                dt_label = self.y_pred[ind]
                for lab in dt_label:
                    fpr_list[int(lab[5])] += 1
                    fpr_score_list[int(lab[5])].append(lab[4])
        fpr_score_stats = []
        for p in fpr_score_list:
            if p:
                fpr_score_stats.append([np.round(np.min(p), 2), np.round(np.median(p), 2), np.round(np.max(p), 2)])
            else:
                fpr_score_stats.append([-1, -1, -1])
        
        # Formalizing results
        res_all = []
        for i, (fn_cnt, fp_cnt, fp_val) in enumerate(zip(fnr_list, fpr_list, fpr_score_stats)):
            res = dict()
            res['class'] = label_map[i]
            res['fn_num'] = fn_cnt
            res['fp_num'] = fp_cnt
            res['fp_min_score'] = fp_val[0]
            res['fp_median_score'] = fp_val[1]
            res['fp_max_score'] = fp_val[2]
            res_all.append(res)
        return res_all

    def draw_failure_cases(self, img_path_list, fn_index_list, fp_index_list, label_map=None, show_scores=False, res_dir=None, group_flag=True):
        """保存failure cases，含漏报和误报的。其中红框为detection的结果，绿框为ground truth的结果。
        Args:
            img_path_list: 图片路径
            fn_index_list: 漏报的样本索引
            fp_index_list: 误报的样本索引
            label_map: 一个映射字典, 如{0:'0', 1:'1', 2:'2', 3:'3', 4:'8', 5:'10'}
            res_dir: failure case存放路径
            group_flag: 是否将漏报和误报分组, True则分组，False则不分。
        Return:
            无
        """
        # Step 1. Make dirs.
        if not res_dir:
            raise ValueError('Result directory error!')
        res_dir_fn = os.path.join(res_dir, 'loubao')
        res_dir_fp = os.path.join(res_dir, 'wubao')
        if not os.path.exists(res_dir_fn):
            os.makedirs(res_dir_fn)
        if not os.path.exists(res_dir_fp):
            os.makedirs(res_dir_fp)
        
        # Step 2. Draw failures.
        if not fn_index_list:
            print('fn_index_list is empty!')
        else:
            for ind in fn_index_list:
                gt_label = self.y_true[ind]
                dt_label = self.y_pred[ind]
                img = mpimg.imread(img_path_list[ind])
                plt.imshow(img)
                currentAxis = plt.gca()
                show_gt_labels, show_dt_labels = [], []
                for lab in gt_label:
                    x1, y1, x2, y2 = lab[0:4]
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
                    currentAxis.add_patch(rect)
                    show_gt_labels.append(label_map[lab[5]])
                for lab in dt_label:
                    x1, y1, x2, y2 = lab[0:4]
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                    currentAxis.add_patch(rect)
                    show_dt_labels.append(label_map[lab[5]])
                if not show_gt_labels:
                    show_gt_labels = ['ok']
                if not show_dt_labels:
                    show_dt_labels = ['ok']
                if show_scores:
                    plt.title('cls:{}->cls:{}\nscore:{}'.format(show_gt_labels, show_dt_labels, self.y_pred_multilabel[ind]))
                else:
                    plt.title('cls:{}->cls:{}'.format(show_gt_labels, show_dt_labels))
                plt.axis('off')
                if not group_flag:
                    output_path = os.path.join(res_dir_fn, os.path.basename(img_path_list[ind]))
                else:
                    # 如果有多个真实标签，则只算其中1个标签
                    res_dir_fn_new = os.path.join(res_dir_fn, show_gt_labels[0])
                    if not os.path.exists(res_dir_fn_new):
                        os.makedirs(res_dir_fn_new)
                    output_path = os.path.join(res_dir_fn_new, os.path.basename(img_path_list[ind]))
                plt.savefig(output_path)
                plt.close()

        if not fp_index_list:
            print('fp_index_list is empty!')
        else:
            for ind in fp_index_list:
                gt_label = self.y_true[ind]
                dt_label = self.y_pred[ind]
                img = mpimg.imread(img_path_list[ind])
                plt.imshow(img)
                currentAxis = plt.gca()
                show_gt_labels, show_dt_labels = [], []
                show_dt_scores = []
                for lab in gt_label:
                    x1, y1, x2, y2 = lab[0:4]
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
                    currentAxis.add_patch(rect)
                    show_gt_labels.append(label_map[lab[5]])
                for lab in dt_label:
                    x1, y1, x2, y2 = lab[0:4]
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                    currentAxis.add_patch(rect)
                    show_dt_labels.append(label_map[lab[5]])
                    show_dt_scores.append(np.round(lab[4], 2))
                if not show_gt_labels:
                    show_gt_labels = ['ok']
                if not show_dt_labels:
                    show_dt_labels = ['ok']
                if show_scores:
                    plt.title('cls:{}->cls:{}\nscore:{}'.format(show_gt_labels, show_dt_labels, show_dt_scores))
                else:
                    plt.title('cls:{}->cls:{}'.format(show_gt_labels, show_dt_labels))
                plt.axis('off')
                if not group_flag:
                    output_path = os.path.join(res_dir_fp, os.path.basename(img_path_list[ind]))
                else:
                    # 如果有多个误报标签，则只算其中1个标签
                    res_dir_fp_new = os.path.join(res_dir_fp, show_dt_labels[0])
                    if not os.path.exists(res_dir_fp_new):
                        os.makedirs(res_dir_fp_new)
                    output_path = os.path.join(res_dir_fp_new, os.path.basename(img_path_list[ind]))
                plt.savefig(output_path)
                plt.close()

    def get_class_ids(self):
        """获取ground_truth中的所有类别id。
        """
        key_list = list(self.gts.keys())
        cls_ids = [cls_id for (img_id, cls_id) in key_list]
        cls_ids_sort = sorted(list(set(cls_ids)))

        key_list_dt = list(self.dts.keys())
        cls_ids_dt = [cls_id for (img_id, cls_id) in key_list_dt]
        cls_ids_dt_sort = sorted(list(set(cls_ids_dt)))

        # 判断检测出的类别是否都在预定义的ground truth类别中。
        for cls_id in cls_ids_dt_sort:
            if cls_id not in cls_ids_sort:
                raise ValueError('检测类别集合为: {}, 真实类别集合为: {}. 检测类别集合不是真实类别集合的子集!'.format(
                    cls_ids_dt_sort, cls_ids_sort))
        return cls_ids_sort

    def get_image_ids(self):
        """获取所有图片的id。
        """
        image_ids = []
        for img_id, _ in enumerate(self.y_true):
            image_ids.append(img_id)
        return image_ids

    def get_gt_inst_num(self):
        """获取各个类别ground_truth实例数目。
        """
        gt_keys = list(self.gts.keys())
        inst_num_dict = dict()
        for cls_id in self.class_ids:
            inst_num_dict[cls_id] = 0
        for img_id, cls_id in gt_keys:
            cnt = len(self.gts[img_id, cls_id])
            inst_num_dict[cls_id] += cnt
        return inst_num_dict

    def reorganize_result(self):
        """重新组织检测结果，便于后续索引。注意这里只整理有结果的图片。
        """
        gts = defaultdict(list)
        dts = defaultdict(list)
        for img_id, (y_true_per_img, y_pred_per_img) in enumerate(zip(self.y_true, self.y_pred)):
            if y_true_per_img:
                for i, true_box in enumerate(y_true_per_img):
                    res = dict()
                    res['inst_id'] = i
                    res['bbox'] = true_box[:4]
                    res['score'] = 1
                    if self.binary_mode:
                        res['cls_id'] = 1
                    else:
                        res['cls_id'] = int(true_box[5])
                    res['img_id'] = int(img_id) # 人为添加，方便后面计算
                    gts[res['img_id'], res['cls_id']].append(res)
            if y_pred_per_img:
                for i, pred_box in enumerate(y_pred_per_img):
                    res = dict()
                    res['inst_id'] = i
                    res['bbox'] = pred_box[:4]
                    res['score'] = pred_box[4]
                    if self.binary_mode:
                        res['cls_id'] = 1
                    else:
                        res['cls_id'] = int(pred_box[5])
                    res['img_id'] = int(img_id) # 人为添加，方便后面计算
                    dts[res['img_id'], res['cls_id']].append(res)
        return gts, dts

    def compute_iou_mat(self, box_arr_1, box_arr_2):
        """计算两组bbox的iou。
        Args:
            box_arr_1:  (N * 4)的array， box形式：[x1, y1, x2, y2]
            box_arr_2:  (M * 4)的array， box形式：[x1, y1, x2, y2]
        Return:
            (N * M)维iou矩阵
        """
        xx_min = np.maximum(box_arr_1[:, 0].reshape(
            [-1, 1]), box_arr_2[:, 0].reshape([1, -1]))  # N*1, M*1 -> N * M
        yy_min = np.maximum(box_arr_1[:, 1].reshape(
            [-1, 1]), box_arr_2[:, 1].reshape([1, -1]))
        xx_max = np.minimum(box_arr_1[:, 2].reshape(
            [-1, 1]), box_arr_2[:, 2].reshape([1, -1]))
        yy_max = np.minimum(box_arr_1[:, 3].reshape(
            [-1, 1]), box_arr_2[:, 3].reshape([1, -1]))

        w = np.maximum(xx_max - xx_min + 1., 0.)
        h = np.maximum(yy_max - yy_min + 1., 0.)
        inter = w * h
        union = ((box_arr_1[:, 2] - box_arr_1[:, 0] + 1) * (box_arr_1[:, 3] - box_arr_1[:, 1] + 1)).reshape([-1, 1]) + (
            (box_arr_2[:, 2] - box_arr_2[:, 0] + 1) * (box_arr_2[:, 3] - box_arr_2[:, 1] + 1)).reshape([1, -1]) - inter
        iou = inter / union
        return iou

    def eval_an_img_on_a_cls(self, img_id, cls_id):
        """评估单张图片在某个类别上的效果。
        Args:
            img_id:  图片id
            cls_id:  类别id
        Return:
            dt_match: 检测出的所有框在gt上的match情况
            dt_match_score: 检测出的所有框在该类别上的score值
            gt_match: gt框在检测出的框上的match情况
            gt_match_score: gt框在该类别上的score值
        """

        dt = self.dts[img_id, cls_id]
        gt = self.gts[img_id, cls_id]

        # 用于计算fn，默认值为-1，表示没有匹配到。
        gt_match = -np.ones(len(gt), dtype=np.int) 
        # 用于计算tp和fp，默认值为-1，表示没有匹配到。 
        dt_match = -np.ones(len(dt), dtype=np.int)  
        gt_match_score = np.zeros(len(gt), dtype=np.float) 
        dt_match_score = np.zeros(len(dt), dtype=np.float)  

        if len(dt) == 0 and len(gt) == 0:
            return [], [], [], []
        elif len(dt) == 0 and len(gt) != 0:
            gt_match_score = [g['score'] for g in gt]
            return [], [], list(gt_match), list(gt_match_score)
        elif len(dt) != 0 and len(gt) == 0:
            dt_match_score = [d['score'] for d in dt]
            return list(dt_match), list(dt_match_score), [], []
        else:
            dt_box = [d['bbox'] for d in dt]  # N * 4
            gt_box = [g['bbox'] for g in gt]  # M * 4
            dt_match_score = [d['score'] for d in dt]
            iou_mat = self.compute_iou_mat(
                np.array(dt_box), np.array(gt_box))  # N * M

            for d_ind, _ in enumerate(dt):
                max_iou_dt = np.max(iou_mat[d_ind, :])
                max_iou_gt_ind = np.argmax(iou_mat[d_ind, :])
                if max_iou_dt >= self.iou:
                    max_iou_dt_ind = np.argmax(iou_mat[:, max_iou_gt_ind])
                    if max_iou_dt_ind == d_ind:
                        dt_match[d_ind] = max_iou_gt_ind
                        gt_match[max_iou_gt_ind] = max_iou_dt_ind # 暂时不用

            return list(dt_match), list(dt_match_score), list(gt_match), list(gt_match_score)

    def collect_results_per_class(self):
        """获取每个类别的所有检测结果。

        Args：
            无
        Return:
            dt_res_all: 每个类别上的所有检测结果
            gt_res_all: 每个类别上的所有gt
        """
        dt_res_all = dict()  # 用于计算tp，fp
        gt_res_all = dict()  # 保留，暂时不用
        for cls_id in self.class_ids:
            dt_res_all[cls_id] = []
            gt_res_all[cls_id] = []
        dt_keys = list(self.dts.keys())
        for img_id, cls_id in dt_keys:
            dt_match, dt_match_score, gt_match, gt_match_score = self.eval_an_img_on_a_cls(img_id, cls_id)
            # 获取dt res，针对某个类别的一个框只存两个东西，score值和是否被检测到的标志。
            if dt_match:
                for _, (dm, dms) in enumerate(zip(dt_match, dt_match_score)):
                    res = dict() 
                    if dm != -1: 
                        res['gt'] = 1 
                    else:
                        res['gt'] = 0
                    res['score'] = dms
                    dt_res_all[cls_id].append(res)
            # 获取gt res，暂时不用，先留着
            if gt_match: 
                for _, (gm, gms) in enumerate(zip(gt_match, gt_match_score)):
                    res = dict() 
                    if gm != -1:
                        res['dt'] = 1 
                    else:
                        res['dt'] = 0
                    res['score'] = gms
                    gt_res_all[cls_id].append(res)
        return dt_res_all, gt_res_all

    def collect_results_per_image(self):
        """获取每张图片的所有检测结果。
        
        Args：
            无
        Return:
            dt_res_all: 每张图片的所有检测结果
        """
        dt_res_all = dict() 
        for img_id in self.image_ids:
            dt_res_all[img_id] = []
        dt_keys = list(self.dts.keys())
        for img_id, cls_id in dt_keys:
            dt_match, _, gt_match, _ = self.eval_an_img_on_a_cls(img_id, cls_id)
            # 情况1： 同时有dt box和gt box
            if dt_match and gt_match:
                for _, dm in enumerate(dt_match): 
                    if dm != -1: 
                        res = 1
                    else:
                        res = 0
                    dt_res_all[img_id].append(res)
            # 情况2： 只有dt box，没有gt box
            elif dt_match and not gt_match: 
                for _, _ in enumerate(dt_match): 
                    dt_res_all[img_id].append(0)
            # 情况3： 没有dt box，dt_res_all[img_id]使用默认值[]

        return dt_res_all

    def compute_precision_and_recall(self, is_norm=True):
        """计算每个类别的precision和recall。
        Args：
            is_norm:  False时不做处理，True时将pr曲线变为单调递减的曲线。
        Return：
            res_list: 所有类别的结果，单个类别结果包含：precision，recall和cls_id
        """
         # 已经把gt包到dt_res中了，计算ap不需要再用到gt_res
        dt_res, _ = self.collect_results_per_class() 
        res_list = []
        for cls_id in self.class_ids:
            res = dict()
            dt_res_sort = sorted(dt_res[cls_id], key=lambda x: x['score'], reverse=True)
            dt_label_arr = np.array([res['gt'] for res in dt_res_sort])
            tps = np.logical_and(dt_label_arr, np.ones(dt_label_arr.shape[0]))
            fps = np.logical_not(dt_label_arr)
            tp_sum = np.cumsum(tps).astype(np.float)
            fp_sum = np.cumsum(fps).astype(np.float)
            
            # 计算原始precision和recall
            precision, recall = [], []
            for tp, fp in zip(tp_sum, fp_sum):
                precision.append(tp / (tp + fp))
                recall.append(tp / self.gt_inst_num[cls_id])
            recall = np.array(recall)
            precision = np.array(precision)

            # 合并recall相同的点, 注意python的set会把原先排好序的recall打乱，要重新排序
            recall_uni = np.sort(np.array(list(set(recall)))) 
            precision_uni = np.zeros(recall_uni.shape)
            for i, rec in enumerate(recall_uni):
                inds = np.where(recall == rec)[0]
                precision_uni[i] = np.max(precision[inds])
            precision, recall = precision_uni, recall_uni

            # 加上头部和尾部的点, 使recall可以到1。
            recall = np.concatenate(([0.], recall, [1.]))
            precision = np.concatenate(([0.], precision, [0.]))

            # 平滑曲线，使其单调递减。这样做AP的计算会稍高一点点，并且曲线也会更好看。
            if is_norm:
                for i in range(precision.shape[0] - 1, 0, -1):
                    precision[i - 1] = np.maximum(precision[i - 1], precision[i])

            # 收集precision和recall结果， recall离散点数为正样本个数
            res['class_id'] = cls_id
            res['precision'] = precision 
            res['recall'] = recall 
            res_list.append(res)
        return res_list


if __name__ == '__main__':
    pass