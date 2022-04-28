
"""atesi EL

Detection + Detection
"""
#
# A demonstration of model ensemble (ME), including:
# 1) ME hyper-parameter optimization
# 2) application of the model ensemble on single images
#

import os, sys
sys.path.append('model_ensemble')
from me_api import import_me_classes
ModelEnsemble, ParameterClass = import_me_classes(mode='detection')
from convert_format import load_pickle, dump_pickle, convert_format_lcz


def main():
    ##### Step 1. 输入数据准备 #####   
    print('##### Step 1. 输入数据准备 #####')
    # 输入相关参数设置
    dataset = '20210915_me'
    input_dir = f'./input/{dataset:s}'
    class_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    model1 = 'yolo'
    model2 = 'ssd'

    # 输出相关参数设置
    output_dir = f'./output/{dataset:s}'
    pickle_path = output_dir + '/optim_param.pkl'

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

    ##### Step 2. 在验证集上训练以获取最优超参 #####
    print('##### Step 2. 在验证集上训练以获取最优超参 #####')
    # 初始化优化器
    me_for_optimization = ModelEnsemble()
    me_for_optimization.initialize_optimizer(
        predictions1 = val_pred1,
        predictions2 = val_pred2,
        ground_truth = val_gt
    )
    # 优化器参数设置
    optimal_param = me_for_optimization.maximize(metric='mAP', miss_threshold=0.005, class_names=class_id_list,
                                                 optimizer='bayesian', max_iterations=2)
    # 在命令行显示最优参数
    optimal_param.dump()      
    # 保存最优参数                          
    optimal_param.pickle(pickle_path=pickle_path)

    ##### Step 3. 使用Step 2中获取的最优超参来做模型融合 #####
    print('##### Step 3. 使用Step 2中获取的最优超参来做模型融合 #####')
    me = ModelEnsemble()
    me.initialize_parameter(
        parameter=ParameterClass.from_pickle(pickle_path=pickle_path)
    )

    # 验证集融合
    val_fused_pred = []
    for pred1, pred2 in zip(val_pred1, val_pred2):
        per_pred = me.fuse(pred1, pred2)
        val_fused_pred.append(per_pred)

    # 测试集融合
    ts_fused_pred = []
    for pred1, pred2 in zip(ts_pred1, ts_pred2):
        per_pred = me.fuse(pred1, pred2)
        ts_fused_pred.append(per_pred)

    ##### Step 4. 将融合结果写到磁盘 #####
    print('##### Step 4. 将融合结果写到磁盘 #####')
    # 处理验证结果
    dump_dir = output_dir + '/val'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    val_gt_path_new = os.path.join(dump_dir, 'gt.pkl')
    val_pred1_path_new = os.path.join(dump_dir, 'pred_{}.pkl'.format(model1))
    val_pred2_path_new = os.path.join(dump_dir, 'pred_{}.pkl'.format(model2))
    val_fused_pred_path_new = os.path.join(dump_dir, 'pred_fused.pkl')

    val_gt = convert_format_lcz(val_gt)
    val_pred1 = convert_format_lcz(val_pred1)
    val_pred2 = convert_format_lcz(val_pred2)
    val_fused_pred = convert_format_lcz(val_fused_pred)

    dump_pickle(val_gt, val_gt_path_new)
    dump_pickle(val_pred1, val_pred1_path_new)
    dump_pickle(val_pred2, val_pred2_path_new)
    dump_pickle(val_fused_pred, val_fused_pred_path_new)

    # 处理测试结果
    dump_dir = output_dir + '/test'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    ts_gt_path_new = os.path.join(dump_dir, 'gt.pkl')
    ts_pred1_path_new = os.path.join(dump_dir, 'pred_{}.pkl'.format(model1))
    ts_pred2_path_new = os.path.join(dump_dir, 'pred_{}.pkl'.format(model2))
    ts_fused_pred_path_new = os.path.join(dump_dir, 'pred_fused.pkl')

    ts_gt = convert_format_lcz(ts_gt)
    ts_pred1 = convert_format_lcz(ts_pred1)
    ts_pred2 = convert_format_lcz(ts_pred2)
    ts_fused_pred = convert_format_lcz(ts_fused_pred)

    dump_pickle(ts_gt, ts_gt_path_new)
    dump_pickle(ts_pred1, ts_pred1_path_new)
    dump_pickle(ts_pred2, ts_pred2_path_new)
    dump_pickle(ts_fused_pred, ts_fused_pred_path_new)
    

if __name__ == '__main__':
    main()