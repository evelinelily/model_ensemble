import pickle as pkl
import cv2
from utils.data_processing import load_pickle
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

cls_names = ['0', '1', '2', '3', '8', '10', '1000', '31', '32', '33', '60']

def draw_boxes(img, all_boxes, model, shift, color=(0, 255, 0)):
    for i in range(len(all_boxes)):
        bboxes = all_boxes[i]
        if bboxes[4] > 0.3:
            xmin, ymin, xmax, ymax = [int(i) for i in bboxes[:4]]

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            info = '{}, {}, {}'.format(model, cls_names[bboxes[5]], round(float(bboxes[4]), 2))
            # print(info)
            cv2.putText(img,
                        info, (xmin-100, ymin+shift),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color,
                        thickness=2,
                        lineType=cv2.LINE_AA)
    return img

def main():
    cls_num = 11
    split = 'val'
    save_dir = './vis/' + split
    os.makedirs(save_dir, exist_ok=True)
    model_list = ['gt', 'yolox_s', 'atss', 'fuse']
    color_list = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
    path_gt = 'pkl/20220304/{}/gts.pkl'.format(split)
    path_im = 'pkl/20220304/{}/img_paths.pkl'.format(split)
    # random_color = np.random.randint(0, 255, (len(model_list) - 1, 3))
    # for i in random_color:
    #     color_list.append(tuple(i))
    # print(1111, color_list)

    pred_list = []
    pred_list.append(load_pickle(path_gt))
    for model in model_list[1:]:
        path_pred = 'pkl/{}/{}.pkl'.format(model, split)
        pred_list.append(load_pickle(path_pred))

    img_paths = load_pickle(path_im)
    for im_id in range(len(img_paths))[10:150]:
        im_path = img_paths[im_id].replace('workspace', 'home/liyu/mnt')
        im = cv2.imread(im_path)
        boxes_all_model = []
        for model_id in range(len(pred_list)):
            # print(model_id)
            pred_per_im = pred_list[model_id][im_id]
            im = draw_boxes(im, pred_per_im, model_list[model_id], 20 * model_id, color_list[model_id])
            if len(pred_per_im) == 0:
                print(im_path)

        name = osp.basename(im_path)
        cv2.imwrite(osp.join(save_dir, name), im)

def single():
    im_name = '3_2_0-49_2462_1259_3663-12004110220563'
    cls_num = 11
    split = 'val'
    save_dir = './vis1/' + split
    os.makedirs(save_dir, exist_ok=True)
    model_list = ['gt', 'yolox_s', 'atss', 'fuse']
    color_list = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
    path_gt = 'pkl/20220304/{}/gts.pkl'.format(split)
    path_im = 'pkl/20220304/{}/img_paths.pkl'.format(split)
    # random_color = np.random.randint(0, 255, (len(model_list) - 1, 3))
    # for i in random_color:
    #     color_list.append(tuple(i))
    # print(1111, color_list)

    pred_list = []
    pred_list.append(load_pickle(path_gt))
    for model in model_list[1:]:
        path_pred = 'pkl/{}/{}.pkl'.format(model, split)
        pred_list.append(load_pickle(path_pred))

    img_paths = load_pickle(path_im)
    for im_id in range(len(img_paths))[10:150]:
        im_path = img_paths[im_id].replace('workspace', 'home/liyu/mnt')
        if im_name not in im_path:
            continue

        im = cv2.imread(im_path)
        im_list = []
        plt.figure()
        for model_id in range(len(pred_list)):
            # print(model_id)
            pred_per_im = pred_list[model_id][im_id]
            if model_id == 0:
                shift = 20
            else:
                shift = 70
            im_new = draw_boxes(im.copy(), pred_per_im, model_list[model_id], shift, color_list[model_id])
            if model_id == 0:
                im = im_new
            else:
                im_list.append(im_new)
                plt.subplot(1,3,model_id)
                plt.imshow(im_new)

        name = osp.basename(im_path)
        cv2.imwrite(osp.join(save_dir, name), cv2.hconcat(im_list))


        # plt.savefig(osp.join(save_dir, name))


if __name__ == '__main__':
    single()