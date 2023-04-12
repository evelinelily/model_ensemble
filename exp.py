from me_interface import MatrixEnsemble
import os
import pickle as pkl
from utils.data_processing import load_pickle

def main(status='train'):
    # 参数设置
    cls_num = 11
    split = 'val' if status == 'train' else 'test'
    path_pred_1 = 'pkl/atss/{}.pkl'.format(split)
    path_pred_2 = 'pkl/yolox_s/{}.pkl'.format(split)
    path_gt = 'pkl/20220304/{}/gts.pkl'.format(split)
    fuse_param_path = './output/param_yoloxs.pkl'
    fuse_res_path = 'pkl/fuse/{}.pkl'.format(split)
    optim_config = {'metric': 'mAP', 'optimizer': 'bayesian', 'max_iterations': 100}
    # 执行优化
    me = MatrixEnsemble(fusion_type='detection', class_id_list=list(range(cls_num)))
    if status == 'train':
        me.optimization(path_pred_1, path_pred_2, path_gt, optim_config)
        save_dir = os.path.dirname(fuse_param_path)
        os.makedirs(save_dir, exist_ok=True)
        me.save_param(fuse_param_path)
    else:
        print('fuse_param_path', fuse_param_path)
        me.load_param(fuse_param_path)
        predictions_1 = load_pickle(path_pred_1)
        predictions_2 = load_pickle(path_pred_2)
        predictions_fused = []
        for pred_1, pred_2 in zip(predictions_1, predictions_2):
            pred_fused = me.fuse(pred_1, pred_2)
            predictions_fused.append(pred_fused)

        save_dir = os.path.dirname(fuse_res_path)
        os.makedirs(save_dir, exist_ok=True)
        with open(fuse_res_path, 'wb') as f:
            pkl.dump(predictions_fused, f)

if __name__ == '__main__':
    for sp in ['train', 'test'][1:]:
        main(sp)

