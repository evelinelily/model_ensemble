#
# Evaluator for detection results
#
import numpy as np
from me_utils import voc_ap, compute_ioumat
from data import DetectionData
from .base import Evaluator

class DetectionEvaluator(Evaluator):
    def eval_instance(self, detections, config):
        """ evaluate in object level, return precisions and recalls in different scores """
        img_names = []
        scores = []
        boxes = []
        recs = {}

        assert 'iou_threshold' in config
        gt_class_names = config['class_names']
        dt_class_names = config['class_names'] if 'class_names_pred' not in config else config['class_names_pred']

        # prepare instances
        for img_name in detections.image_names:
            for instance_object in detections[img_name].instance_object_list:
                score = self._compute_instance_score(instance_object.distribution, dt_class_names)
                img_names.append(img_name)
                scores.append(score)
                boxes.append(instance_object.box)

            R = []
            for instance_object in self.annotations[img_name].instance_object_list:
                gt_class_name = self.label2class[instance_object.fg_class_id]
                if gt_class_name in gt_class_names:
                    R.append({'bbox': instance_object.box})
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

    def compute_ap(self, predictions, config):
        """ Compute AP given predictions and config """
        assert isinstance(predictions, DetectionData), "predictions should be DetectionData type."
        assert isinstance(self.annotations, DetectionData), "annotations should be DetectionData type."
        assert self.annotations.compatible_with(predictions), "predictions not match with annotations."

        res = self.eval_instance(detections=predictions, config=config)
        return res['ap']