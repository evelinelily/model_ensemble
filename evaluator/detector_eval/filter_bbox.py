"""
Utility functions to filter bbox in manually.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle as pkl


# cutting box: [280, 590]

def filter_by_index(det_res_fusion, del_index):
    new_det_res_fusion = [[] for _ in range(len(det_res_fusion))]
    for idx, del_idx in enumerate(del_index):
        if del_idx:
            new_res = det_res_fusion[idx]

            for del_num, index in enumerate(del_idx):
                del new_res[index-del_num]
            new_det_res_fusion[idx] = new_res
        else:
            new_det_res_fusion[idx] = det_res_fusion[idx]
    return new_det_res_fusion


def filter_10(det_res_fusion, tolerated_area_ratio=0.025, cut_box=[280, 590]):
    """
    1) 面积小于电池片面积15%的直接去掉
    Args:
        det_res_fusion: fusion model return result.
        tolerated_area_ratio: 
        cut_box: cutting model return result.

    Returns:
        filter result by setting manual param.
    """
    area = cut_box[0] * cut_box[1]
    del_idx = [[] for _ in range(len(det_res_fusion))]
    for det_idx, det_res in enumerate(det_res_fusion):
        if det_res:
            for bbox_idx, det in enumerate(det_res):
                if int(det[-1]) == 5:
                    if (det[2] - det[0]) * (det[3] - det[1]) < area * tolerated_area_ratio:
                        del_idx[det_idx].append(bbox_idx)
    new_det_res_fusion = filter_by_index(det_res_fusion, del_idx)
    return new_det_res_fusion


def filter_3(det_res_fusion, define_boundary=0.05, tolerated_ratio=0.25, common_score_th=0.1924, cut_box=[280, 590]):
    """
    1) 靠左右两边且长度小于0.25p，轻微，直接扔掉.
    2) 其他：严重，按统一阈值
    Args:
        det_res_fusion: fusion model return result.
        define_boundary
        tolerated_ratio: midpoint of defect coordinates in the 1x1 panel image width ratio.
        common_score_th
        cut_box: cutting model return result.
        w_tolerated_ratio:

    Returns:
        filter result by setting manual param.
    """
    boundary_width = cut_box[0] * define_boundary
    del_idx = [[] for _ in range(len(det_res_fusion))]
    for det_idx, det_res in enumerate(det_res_fusion):
        if det_res:
            for bbox_idx, det in enumerate(det_res):
                if int(det[-1]) == 3:
                    if det[0] < boundary_width or det[2] > (cut_box[0] - boundary_width):
                        if (det[2]-det[0]) < cut_box[0] * tolerated_ratio:
                            del_idx[det_idx].append(bbox_idx)
                    elif det[4] < common_score_th:
                        del_idx[det_idx].append(bbox_idx)

    new_det_res_fusion = filter_by_index(det_res_fusion, del_idx)

    return new_det_res_fusion


def filter_2(det_res_fusion, define_boundary=0.05, tolerated_len_ratio=0.15, common_score_th=0.1924, cut_box=[280, 590]):
    """
    1) 不靠四边：轻微，长度小于0.025 piece_area.
    2) 靠四边：严重，按统一阈值.
    Args:
        det_res_fusion: fusion model return result.
        define_boundary
        common_score_th
        cut_box
        tolerated_ratio:
        tolerated_area_ratio

    Returns:

    """
    del_idx = [[] for _ in range(len(det_res_fusion))]
    for det_idx, det_res in enumerate(det_res_fusion):
        if det_res:
            for bbox_idx, det in enumerate(det_res):
                if int(det[-1]) == 2:
                    # check detect res not in the side of the image
                    if not (det[0] < define_boundary * cut_box[0] or det[2] > (cut_box[0] - define_boundary * cut_box[0]) or det[1] < define_boundary * cut_box[1] or det[3] > (cut_box[1] - define_boundary * cut_box[1])):
                        defect_len = max((det[2] - det[0]), (det[3] - det[1]))
                        if defect_len < cut_box[1] * tolerated_len_ratio:
                            del_idx[det_idx].append(bbox_idx)
                    elif det[4] < common_score_th:
                        del_idx[det_idx].append(bbox_idx)

    new_det_res_fusion = filter_by_index(det_res_fusion, del_idx)
    return new_det_res_fusion


def filter_0(det_res_fusion, define_corner=0.1, tolerated_ratio=0.1, cut_box=[280, 590], score_th=0.4, common_score_th=0.1924):
    """
    1) 缺角破片（4个角上的破片）：面积小于(0.1p * 0.1p), 过滤
    2) 其他破片：按统一阈值
    Args:
        det_res_fusion: fusion model return result.
        define_corner
        tolerated_ratio:
        cut_box
        common_score_th
    Returns:

    """
    del_idx = [[] for _ in range(len(det_res_fusion))]
    for det_idx, det_res in enumerate(det_res_fusion):
        if det_res:
            for bbox_idx, det in enumerate(det_res):
                if int(det[-1]) == 0:
                    appear_in_corner = False
                    corner_boundary = cut_box[1] * define_corner
                    # top-left corner
                    if det[0] < corner_boundary and det[1] < corner_boundary:
                        appear_in_corner = True
                    # bottom-left
                    if det[0] < corner_boundary and det[3] > cut_box[1] - corner_boundary:
                        appear_in_corner = True
                    # top-right corner
                    if det[2] > cut_box[0] - corner_boundary and det[1] < corner_boundary:
                        appear_in_corner = True
                    # bottom-right corner
                    if det[2] > cut_box[0] - corner_boundary and det[3] > cut_box[1] - corner_boundary:
                        appear_in_corner = True
                    # detect defect in the corner.
                    # if appear_in_corner and (det[2]-det[0]) * (det[3]-det[1]) < pow(cut_box[1]*tolerated_ratio, 2):
                    if appear_in_corner and (det[2]-det[0]) * (det[3]-det[1]) < pow(cut_box[1]*tolerated_ratio, 2) and det[4] < score_th:
                        del_idx[det_idx].append(bbox_idx)
                    elif det[4] < common_score_th:
                        del_idx[det_idx].append(bbox_idx)

    new_det_res_fusion = filter_by_index(det_res_fusion, del_idx)
    return new_det_res_fusion


def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as fid:
        return pkl.load(fid)


def dump_pickle(res, pickle_file):
    with open(pickle_file, 'wb') as fid:
        return pkl.dump(res, fid)


if __name__ == "__main__":
    filter_path = '/disk/gitlab/lujuntao/region_classification/result/test_pkl/fused_result_0817/test/fused.pkl'
    res_path = '/disk/gitlab/lujuntao/region_classification/result/test_pkl/fused_result_0817/test/filter_0817_v2.pkl'

    det_res_fusion = load_pickle(filter_path)

    det_res_fusion = filter_0(det_res_fusion, define_corner=0.1, tolerated_ratio=0.1, cut_box=[280, 590],
                              common_score_th=0.1924)

    det_res_fusion = filter_2(det_res_fusion, define_boundary=0.05, tolerated_area_ratio=0.02, common_score_th=0.1924,
                              cut_box=[280, 590])

    det_res_fusion = filter_3(det_res_fusion, define_boundary=0.05, tolerated_ratio=0.25, common_score_th=0.1924,
                              cut_box=[280, 590])

    dump_pickle(det_res_fusion, pickle_file=res_path)