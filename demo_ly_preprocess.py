import os
from convert_format import load_pickle, dump_pickle, convert_format_me, filter_out_result

if __name__ == '__main__':
    # 输入相关参数
    dataset = '20210915'
    input_dir = f'./input/{dataset:s}'
    # class_names = ['0', '1', '2', '3', '8', '10', '1000', '31', '33', '60']
    class_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    model1 = 'yolo'
    model2 = 'ssd'
    filter_score_th = 0.05

    # 加载待融合数据
    val_gt_path = input_dir + '/val/gt.pkl'
    val_pred1_path = input_dir + f'/val/pred_{model1:s}.pkl'
    val_pred2_path = input_dir + f'/val/pred_{model2:s}.pkl'
    ts_gt_path = input_dir + '/test/gt.pkl'
    ts_pred1_path = input_dir + f'/test/pred_{model1:s}.pkl'
    ts_pred2_path = input_dir + f'/test/pred_{model2:s}.pkl'
    val_gt = load_pickle(val_gt_path)
    val_pred1 = load_pickle(val_pred1_path)
    val_pred2 = load_pickle(val_pred2_path)
    ts_gt = load_pickle(ts_gt_path)
    ts_pred1 = load_pickle(ts_pred1_path)
    ts_pred2 = load_pickle(ts_pred2_path)

    # 对待融合数据进行预处理
    val_pred1 = filter_out_result(val_pred1, score=filter_score_th)
    val_pred2 = filter_out_result(val_pred2, score=filter_score_th)
    ts_pred1 = filter_out_result(ts_pred1, score=filter_score_th)
    ts_pred2 = filter_out_result(ts_pred2, score=filter_score_th)

    # 对待融合数据做格式转换
    val_gt = convert_format_me(val_gt, class_id_list)
    val_pred1 = convert_format_me(val_pred1, class_id_list)
    val_pred2 = convert_format_me(val_pred2, class_id_list)
    ts_gt = convert_format_me(ts_gt, class_id_list)
    ts_pred1 = convert_format_me(ts_pred1, class_id_list)
    ts_pred2 = convert_format_me(ts_pred2, class_id_list)

    # 将me格式的数据写回磁盘
    input_dir_new_val = f'./input/{dataset:s}_me/val'
    input_dir_new_test = f'./input/{dataset:s}_me/test'
    if not os.path.exists(input_dir_new_val):
        os.makedirs(input_dir_new_val)
    if not os.path.exists(input_dir_new_test):
        os.makedirs(input_dir_new_test)
    val_gt_path_new = input_dir_new_val + '/gt.pkl'
    val_pred1_path_new = input_dir_new_val + f'/pred_{model1:s}.pkl'
    val_pred2_path_new = input_dir_new_val + f'/pred_{model2:s}.pkl'
    ts_gt_path_new = input_dir_new_test + '/gt.pkl'
    ts_pred1_path_new = input_dir_new_test + f'/pred_{model1:s}.pkl'
    ts_pred2_path_new = input_dir_new_test + f'/pred_{model2:s}.pkl'
    dump_pickle(val_gt, val_gt_path_new)
    dump_pickle(val_pred1, val_pred1_path_new)
    dump_pickle(val_pred2, val_pred2_path_new)
    dump_pickle(ts_gt, ts_gt_path_new)
    dump_pickle(ts_pred1, ts_pred1_path_new)
    dump_pickle(ts_pred2, ts_pred2_path_new)