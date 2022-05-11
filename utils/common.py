import numpy as np

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


def nms(boxes, scores=None, overlapThresh=.3):
    boxes = np.array(boxes, dtype=np.float32).reshape([-1,4])
    scores = scores if scores is None else np.array(scores, dtype=np.float32)

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # if scores are given, sort boxes such that nms keep high score boxes first,
    # otherwise sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    if scores is None:
        idxs = np.argsort(y2)
    else:
        idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        inters = w * h
        iou = inters / (area[i]+area[idxs[:last]]-inters)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > overlapThresh)[0])))

    return pick