#
# Manage json files that store detection results or ground truths
#
import json
import numpy as np

def apply_distribution_weight(distribution):
    """ apply weights to distribution """
    assert len(distribution) == 1+7
    class_weights = [1.] + [1.,1.] + [1.]*5
    new_distribution = [s*w for s,w in zip(distribution, class_weights)]
    # normalize the new distribution
    new_distribution_sum = sum(new_distribution)
    new_distribution = [s/new_distribution_sum for s in new_distribution]
    return new_distribution

class JsonManager(object):
    def __init__(self, json_file):
        self.image_dict, self.label2class = self.parse_json(json_file)

    @property
    def size(self):
        return len(self.image_dict)

    @property
    def instance_size(self):
        return sum(len(x) for x in self.image_dict.values())

    @property
    def instance_num(self):
        return self.instance_size

    @property
    def image_num(self):
        return self.size

    def filter_instances(self, score_threshold):
        """ filter out instances with a score threshold """
        for img_name in self.iterator():
            new_instance_list = list()
            for instance in self.image_dict[img_name]:
                assert instance['distribution'] is not None, "filter_instances() only applies in detection JsonManager object"
                max_fg_cls_prob = np.max(instance['distribution'][1:])
                if max_fg_cls_prob >= score_threshold:
                    new_instance_list.append(instance)
            self.image_dict[img_name] = new_instance_list


    def get_instance_num(self, class_names):
        counter = 0
        for image_instances in self.image_dict.values():
            for instance in image_instances:
                if instance['class_name'] in class_names:
                    counter += 1
        return counter

    def __getitem__(self, key):
        return self.image_dict[key]

    def __setitem__(self, key, value):
        self.image_dict[key] = value

    def iterator(self):
        for image_name in self.image_dict:
            yield image_name

    def parse_json(self, json_file):
        with open(json_file, 'r') as fid:
            json_obj = json.load(fid)

        # deal with class dict
        class_dict = {}
        for class_obj in json_obj['class_dict']:
            class_dict[class_obj['class_id']] = class_obj['class_name']

        image_dict = dict()
        for item in json_obj['image']:
            image_dict[item['image_id']] = item

        instance_dict = dict()
        for item in json_obj['instance']:
            image_id = item['image_id']
            image_name = image_dict[image_id]['image_name']
            x, y, w, h = item['instance_bbox']
            assert w > 0 and h > 0, "Invalid bbox found in {}".format(image_name)

            # check whether prob distribution of the instance exists (found only in detection objection)
            if 'instance_scores' in item:
                distribution = item['instance_scores']
                # distribution = apply_distribution_weight(distribution)   # this function is class_dict sensitive, disable it
                class_id = np.argmax(distribution[1:])+1  # get class id from fg distribution: argmax(distribution[:1])+1
                # assert class_id == item['class_id'], "{} vs. {}".format(class_id, item['class_id'])
            else:
                distribution = None
                class_id = item['class_id']

            content = {
                'class_name': class_dict[class_id],
                'box': (x, y, x+w, y+h),
                'distribution': distribution
                }
            if image_name in instance_dict:
                instance_dict[image_name].append(content)
            else:
                instance_dict[image_name] = [content]

        image_dict = dict()
        for item in json_obj['image']:
            image_name = item['image_name']
            assert image_name not in image_dict, 'Duplicated image found: {}'.format(image_name)
            if image_name in instance_dict:
                image_dict[image_name] = instance_dict[image_name]
            else:
                image_dict[image_name] = [] # assign empty list if no instance is found

        return image_dict, class_dict

    def issubset(self, json_obj):
        """ check if another json file object focuses on the same image source """
        assert isinstance(json_obj, JsonManager)

        # check if class_dict equal (dictionary allows == sign)
        if not self.label2class == json_obj.label2class:
            return False

        # check if our samples covered by the input json object samples
        return set(self.image_dict.keys()).issubset(set(json_obj.image_dict.keys()))

    def equal(self, json_obj):
        """ check if another json file object exactly focuses on the same image source """
        return self.issubset(json_obj) and json_obj.issubset(self)

    def collect_instance_info(self):
        """ count number of instance per class """
        instance_info = dict()
        for instances in self.image_dict.values():
            for inst in instances:
                class_name = inst['class_name']
                if class_name in instance_info:
                    instance_info[class_name] += 1
                else:
                    instance_info[class_name] = 1
        return instance_info

    def dump_info(self):
        instance_info = self.collect_instance_info()
        print(instance_info, "in {} images".format(self.size))


    @staticmethod
    def filter_single_vi_object(detections, annotations):
        """ remove vi boxes that are not detected by AI pipeline from both the
        detections and annotations
        """
        assert detections.issubset(annotations) and annotations.issubset(detections)

        def _is_single_vi(dist):
            D = np.array(dist, dtype=np.float32)
            return np.abs(D[:-1] - D[1:]).mean() < 1e-8

        for img_name in detections.iterator():
            contents_det = detections[img_name]
            contents_ann = annotations[img_name]
            contents_det_clean, contents_ann_clean = list(), list()
            for cont_det, cont_ann in zip(contents_det, contents_ann):
                assert all([x==y for x,y in zip(cont_det['box'], cont_ann['box'])])  # assert the box saving order the same in detection and annotation
                if not _is_single_vi(cont_det['distribution']):
                    contents_det_clean.append(cont_det)
                    contents_ann_clean.append(cont_ann)
                # else:
                #     print(cont_det['distribution'])
            # update detections and annotations of current image
            detections[img_name] = contents_det_clean
            annotations[img_name] = contents_ann_clean


    def convert_miss_to_qita(self):
        """ convert missing instance (vi boxes that are not detected by AI pipeline, which are indicated by their uniform distribution)
        to 'qita' class:
        1) add '0: qita' to self.label2class
        2) change the 'class_name' value of missing instances to 'qita'
        """
        # function to determine whether an instance is missed by AI (from its distribution)
        def _is_missing_instance(dist):
            D = np.array(dist, dtype=np.float32)
            return np.abs(D[:-1] - D[1:]).mean() < 1e-8

        # 1) add '0: qita' to self.label2class
        assert 0 not in self.label2class, "key 0 already in self.label2class"
        self.label2class[0] = 'qita'

        # change the 'class_name' value of missing instances to 'qita'
        for instance_list in self.image_dict.values():
            for content in instance_list:
                if content['distribution'] is not None: # only detection instance are changed
                    if _is_missing_instance(content['distribution']):
                        content['class_name'] = 'qita'  # this change will happened in place

    @classmethod
    def init_as_classification(cls, json_file):
        """ initialized as classification prediction/groundtruth, used in Evaluator4Classifier """
        jm = cls(json_file)
        jm.convert_miss_to_qita()
        return jm


if __name__ == '__main__':
    jm_det = JsonManager('/mnt/mt4_tmp/zhaimenghua/zhaimenghua/experiments/results/preds-low_rank_first/baseline/large-vi/cls.json')
    jm_gth = JsonManager('/mnt/mt4_tmp/zhaimenghua/zhaimenghua/experiments/results/gths/large/vi.json')

    print(jm_det.equal(jm_gth))