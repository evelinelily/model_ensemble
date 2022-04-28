import numpy as np
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from .confusion_matrix_pretty_print import pretty_plot_confusion_matrix

def draw_confusion_matrix(det_labels, gth_labels, save_path='./confusion_matrix.png'):
    '''
    compute and draw confusion matrix
    params:
        det_labels: detection labels (a list of string)
        gth_labels: annotation labels (a list of string)
        save_path: path to save the confusion matrix
    note: det_labels and gth_labels must have an element-wise correspondency
    '''
    assert len(det_labels) == len(gth_labels)
    all_labels = list(set(gth_labels).union(set(det_labels)))
    all_labels = sorted(all_labels)
    cm = confusion_matrix(gth_labels, det_labels, labels=all_labels)

    df_cm = DataFrame(cm, index=all_labels, columns=all_labels)
    cmap = 'PuRd'
    fig_cm = pretty_plot_confusion_matrix(df_cm, cmap=cmap)
    fig_cm.savefig(save_path)


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_ioumat(boxes1, boxes2):
    '''
    Compute IoU matrix between boxes1 and boxes2
    boxes1: shape=Nx4
    boxes2: shape=Mx4
    return iou shape=NxM
    '''
    boxes1 = np.array(boxes1, dtype=np.float32).reshape([-1,4])
    boxes2 = np.array(boxes2, dtype=np.float32).reshape([-1,4])

    ixmin = np.maximum(boxes1[:, 0].reshape([-1, 1]),
                       boxes2[:, 0].reshape([1, -1]))
    iymin = np.maximum(boxes1[:, 1].reshape([-1, 1]),
                       boxes2[:, 1].reshape([1, -1]))
    ixmax = np.minimum(boxes1[:, 2].reshape([-1, 1]),
                       boxes2[:, 2].reshape([1, -1]))
    iymax = np.minimum(boxes1[:, 3].reshape([-1, 1]),
                       boxes2[:, 3].reshape([1, -1]))
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((boxes1[:, 2] - boxes1[:, 0] + 1.) * (boxes1[:, 3] - boxes1[:, 1] + 1.)).reshape([-1, 1]) + \
          ((boxes2[:, 2] - boxes2[:, 0] + 1.) * (boxes2[:, 3] - boxes2[:, 1] + 1.)).reshape([1, -1]) - inters

    iou = inters / uni

    return iou