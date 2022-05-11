from __future__ import absolute_import

import numpy as np
from tqdm import tqdm
import os, collections
from utils import voc_ap, compute_ioumat, draw_confusion_matrix
from utils import JsonManager

class Evaluator(object):
    def __init__(self, det_json, gth_json):
        self.detections = JsonManager(det_json)
        self.annotations = JsonManager(gth_json)
        self.label2class = self.detections.label2class
        raise NotImplementedError

    @property
    def class2label(self):
        return {y:x for x,y in self.label2class.items()}

    @property
    def class_names(self):
        return list(self.label2class.values())

    def filter_detection_results(self, score_threshold):
        """ filter detection results with a score threshold """
        self.detections.filter_instances(score_threshold=score_threshold)

    def _compute_instance_score(self, instance, class_names):
        """ compute instance score given classes """
        distribution = instance['distribution']
        assert len(list(class_names)) == len(set(class_names)), "class_names: '{}' containes duplicated class".format(class_names)
        if distribution is not None:
            class2label = self.class2label
            score = sum(distribution[class2label[cls_name]] for cls_name in class_names)
            return score
        else:
            return len(class_names) / (len(self.label2class)+1)

    def eval_instance_level(self, config):
        ''' evaluate in object level, return precisions and recalls in different scores '''
        img_names = []
        scores = []
        boxes = []
        recs = {}

        gt_class_names = config['class_names']
        dt_class_names = config['class_names'] if 'class_names_pred' not in config else config['class_names_pred']

        # prepare instances
        for img_name in self.detections.iterator():
            for content in self.detections[img_name]:
                score = self._compute_instance_score(content, dt_class_names)
                img_names.append(img_name)
                scores.append(score)
                boxes.append(content['box'])

            R = []
            for content in self.annotations[img_name]:
                if content['class_name'] in gt_class_names:
                    R.append({'bbox': content['box']})
            recs[img_name] = R

        # 1. prepare
        # convert list to np array for better indexing operation
        img_names = np.array(img_names)

        # 2. get gtboxes for this class.
        class_recs = {}
        num_pos = 0
        for imagename, R in recs.items():
            R = recs[imagename]
            bbox = np.array([x['bbox'] for x in R])
            det = [False] * len(R)
            num_pos = num_pos + len(R)
            class_recs[imagename] = {'bbox': bbox,
                                     'det': det}  # det means that gtboxes has already been detected

        # 3. read the detection file
        image_ids = img_names
        confidence = np.array(scores, dtype=np.float32)
        BB = np.array(boxes, dtype=np.float32)
        sorted_scores = confidence.copy()

        nd = len(image_ids)  # num of detections. That, a line is a det_box.
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        if BB.shape[0] > 0:
            # sort by confidence
            sorted_ind = np.argsort(0.-confidence)
            sorted_scores = confidence[sorted_ind]
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]  # reorder the img_name

            # go down dets and mark TPs and FPs
            for d in range(nd):
                R = class_recs[image_ids[d]]  # img_id is img_name
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)

                if BBGT.size > 0:
                    # compute overlaps
                    overlaps = compute_ioumat(BBGT, bb)
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > config['iou_threshold']:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.

        # 4. get recall, precison and AP
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / np.maximum(float(num_pos), np.finfo(float).eps)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(float).eps)
        ap = voc_ap(rec, prec, use_07_metric=False)

        # complete the beginning and the end of rec and prec (for drawing line only)
        rec = np.hstack([0.,rec,1.])
        prec = np.hstack([1.,prec,0.])
        sorted_scores = np.hstack([1.,sorted_scores,0.])

        return {'recalls': rec, 'precisions': prec, 'ap': ap, 'scores': sorted_scores}

    def eval_image_level(self, config):
        type_counter = {'TP':0, 'FP':0, 'TN':0, 'FN':0}
        gt_class_names = config['class_names']
        dt_class_names = config['class_names'] if 'class_names_pred' not in config else config['class_names_pred']

        for img_name in self.detections.iterator():
            # load detection boxes for the image
            boxes_dt = list()
            for content in self.detections[img_name]:
                score = self._compute_instance_score(content, dt_class_names)
                if score > config['score_threshold']:     # we only focus detections whose scores are greater than the given threshold
                    boxes_dt.append(content['box'])

            # load groundtruth boxes for the image
            boxes_gt = list()
            for content in self.annotations[img_name]:
                if content['class_name'] in gt_class_names:
                    boxes_gt.append(content['box'])
            
            # Determine which type (TP/TN/FP/FN) the image belongs to
            image_type = None
            if len(boxes_gt) == 0:
                if len(boxes_dt) > 0:
                    image_type = 'FP'
                else:
                    image_type = 'TN'
            else:
                ioumat = compute_ioumat(boxes_gt, boxes_dt)
                ovlmat = ioumat > config['iou_threshold']

                if 'mode' not in config or config['mode'] == 'easy':
                    hit_condition = np.any(ovlmat)
                else:
                    hit_condition = np.all(np.any(ovlmat, axis=1))

                if hit_condition:
                    image_type = 'TP'
                else:
                    image_type = 'FN'

            assert image_type is not None, img_name
            type_counter[image_type] += 1

        # compute recall and precision
        res = {
            'TP': type_counter['TP'], 'FP': type_counter['FP'],
            'TN': type_counter['TN'], 'FN': type_counter['FN'],
            'miss':      type_counter['FN'] / max(1, type_counter['TP']+type_counter['FN']),   # miss rate
            'overkill':  type_counter['FP'] / max(1, type_counter['TN']+type_counter['FP']),   # false positive rate (probability of false alarm)
            'recall':    type_counter['TP'] / max(1, type_counter['TP']+type_counter['FN']),   # recall rate (sensitivity)
            'precision': type_counter['TP'] / max(1, type_counter['TP']+type_counter['FP'])    # precision (positive predictive value)
        }

        return res

    def find_miss_instances_function(self, detections, annotations, config):
        """ find all miss-detected annotation instances
            Return: A something like JsonManager.image_dict """
        missed_image_dict = dict()
        gt_class_names = config['class_names']
        dt_class_names = config['class_names'] if 'class_names_pred' not in config else config['class_names_pred']
        # loop over all groundtruth images
        for img_name in annotations.iterator():
            img_instances_det = detections[img_name]
            img_instances_gth = annotations[img_name]
            boxes_dt = [x['box'] for x in img_instances_det]
            boxes_gt = [x['box'] for x in img_instances_gth]
            overlap_mat = compute_ioumat(boxes_gt, boxes_dt) > config['iou_threshold']
            img_instances = list()
            for idx, instance_gth in enumerate(img_instances_gth):
                if instance_gth['class_name'] in gt_class_names:
                    # find all overlapping detection instances
                    overlap_det_ids = np.where(overlap_mat[idx,:])[0]
                    # check if there any overlapping detection instance
                    # has the same class with the ground truth
                    no_detection_hit_groundtruth = True
                    for idy in overlap_det_ids:
                        instance_det = img_instances_det[idy]
                        if instance_det['class_name'] in dt_class_names:
                            no_detection_hit_groundtruth = False
                            break
                    if no_detection_hit_groundtruth:
                        img_instances.append(instance_gth)
            # save the missed groundtruth instances in this image
            if len(img_instances) > 0:
                missed_image_dict[img_name] = img_instances
        return missed_image_dict


    def find_miss_instances(self, config):
        return self.find_miss_instances_function(self.detections, self.annotations, config)

    def find_fa_instances(self, config):
        return self.find_miss_instances_function(self.annotations, self.detections, config)


class Evaluator4Classifier(Evaluator):
    def __init__(self, det_json, gth_json):
        self.detections = JsonManager.init_as_classification(det_json)
        self.annotations = JsonManager.init_as_classification(gth_json)

        # gaurrentee the ground truth json can evaluate the detection json
        assert self.detections.equal(self.annotations), "Detection file and groundtruth file mismatch"

        print("="*70)
        print("{} images and {} instances found on the detection json".format(self.detections.size, self.detections.instance_size))
        self.detections.dump_info()
        print("-"*50)
        print("{} images and {} instances found on the groundtruth json".format(self.annotations.size, self.annotations.instance_size))
        self.annotations.dump_info()
        print("="*70, '\n\n')

        self.label2class = self.detections.label2class

    def draw_confusion_matrix(self, save_path='./confusion_matrix.png'):
        print("drawing confusion matrix...")
        det_labels, ann_labels = list(), list()
        for _, (contents_det, contents_ann) in self.items():
            for cont_det, cont_ann in zip(contents_det, contents_ann):
                det_labels.append(str(self.class2label[cont_det['class_name']])+'-'+cont_det['class_name'])
                ann_labels.append(str(self.class2label[cont_ann['class_name']])+'-'+cont_ann['class_name'])
        draw_confusion_matrix(det_labels, ann_labels, save_path)

    def items(self):
        """ return a iterator that yields tuples: (image_name, (self.detections[img_name], self.annotations[img_name])) """
        assert self.detections.equal(self.annotations)
        for img_name in self.detections.iterator():
            yield img_name, (self.detections[img_name], self.annotations[img_name])



class Evaluator4Detector(Evaluator):
    def __init__(self, det_json, gth_json):
        self.detections = JsonManager(det_json)
        self.annotations = JsonManager(gth_json)

        print("="*70)
        print("{} images and {} instances found on the detection json".format(self.detections.size, self.detections.instance_size))
        self.detections.dump_info()
        print("-"*50)
        print("{} images and {} instances found on the groundtruth json".format(self.annotations.size, self.annotations.instance_size))
        self.annotations.dump_info()
        print("="*70, '\n\n')

        # gaurrentee the ground truth json can evaluate the detection json
        assert self.detections.issubset(self.annotations), "Detection file and groundtruth file mismatch"

        self.label2class = self.detections.label2class

    def classify_each_object(self, config):
        """ return the labels of both groundtruths and detection bboxes """
        label_pairs = []
        for img_name in self.detections.iterator():
            labels_dt, boxes_dt = [], []
            for content in self.detections[img_name]:
                score = self._compute_instance_score(content, [content['class_name']])
                if score > config['score_threshold']:
                    labels_dt.append(content['class_name'])
                    boxes_dt.append(content['box'])

            # collect ground truth
            labels_gt, boxes_gt = [], []
            for content in self.annotations[img_name]:
                labels_gt.append(content['class_name'])
                boxes_gt.append(content['box'])

        # for sample in self.__samples:
        #     img_path = sample.info['img_path']
        #     # collect detections
        #     labels_dt, boxes_dt = [], []
        #     for obj in sample.instance_dt.iterator():
        #         if obj['score'] > config['score_threshold']:
        #             labels_dt.append(obj['label'])
        #             boxes_dt.append(obj['bbox'])

        #     # collect ground truth
        #     labels_gt, boxes_gt = [], []
        #     for obj in sample.instance_gt.iterator():
        #         labels_gt.append(obj['label'])
        #         boxes_gt.append(obj['bbox'])

            # match det and gth
            # only keep the maximum iou for each detect (force each detect to match only one gtbox)
            ioumat = compute_ioumat(boxes_gt, boxes_dt)
            if ioumat.size > 0:
                maxids = ioumat.argmax(axis=0)
                mask = np.zeros_like(ioumat)
                mask[maxids,np.arange(mask.shape[1])] = 1
                pairmat = ioumat * mask > config['iou_threshold']
            else:
                pairmat = ioumat > config['iou_threshold']

            # scan over all boxes (detections, groundtruth togather)
            gth_ids, det_ids = list(range(len(labels_gt))), list(range(len(labels_dt)))
            # loop over ground truth first
            for gid in gth_ids:
                ovlids = np.where(pairmat[gid,:])[0]
                # if find detections that overlap with the ground truth
                if len(ovlids) > 0:
                    for did in ovlids:
                        label_pair = (labels_gt[gid], labels_dt[did])
                        det_ids.remove(did)  # remove matched detection from the inspection list
                else: # if no detection overlap with the ground truth
                    label_pair = (labels_gt[gid], '0-bg')
                label_pairs.append(label_pair)

            # loop over the unmatched detections
            for did in det_ids:
                label_pair = ('0-bg', labels_dt[did])
                label_pairs.append(label_pair)

        # compute confusion matrix
        gth_labels, det_labels = zip(*label_pairs)

        return gth_labels, det_labels

    def draw_confusion_matrix(self, iou_threshold, save_path='./confusion_matrix.png'):
        ''' compute confusion matrix '''
        config = {
            'iou_threshold': iou_threshold,
            'score_threshold': 0,  # keep all bboxes generated by inference
        }
        gth_labels, det_labels = self.classify_each_object(config)
        draw_confusion_matrix(det_labels, gth_labels, save_path)


if __name__ == '__main__':
    ev = Evaluator4Detector(
        det_json = 'jsons/3L-new/exp6/cls.json',
        gth_json = 'jsons/3L-new/annotations.json'
        )
    config = {
        'iou_threshold': .1,
        'score_threshold': .5,
        'class_names': ['wuzi', 'lianjiao']
        }
    res = ev.find_fa_instances(config)
    # print(res)