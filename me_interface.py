"""MatrixEnsemble接口类"""

import os
import sys

from utils.data_processing import load_pickle, dump_pickle, filter_scores, convert_format_me
from me_methods import get_me_classes


### 指定融合模式接口函数 ###
class MatrixEnsemble(object):

    def __init__(self, fusion_type, class_id_list=None):
        self.fusion_type = fusion_type
        self.class_id_list = class_id_list
        self.ModelEnsemble, self.ParameterClass = get_me_classes(self.fusion_type)
        self.model_ensemble = self.ModelEnsemble()
        self.optimal_param = None

    def optimization(self, path_pred_1, path_pred_2, path_gt, optim_config):
        """执行优化操作，获取最优融合超参数。
        """
        # 参数补全
        if not optim_config:
            optim_config['metric'] = 'mAP'
            optim_config['optimizer'] = 'bayesian'
            optim_config['max_iterations'] = 100
            optim_config['miss_threshold'] = 0
        else:
            if 'metric' not in optim_config:
                optim_config['metric'] = 'mAP'
            if 'optimizer' not in optim_config:
                optim_config['optimizer'] = 'bayesian'
            if 'max_iterations' not in optim_config:
                optim_config['max_iterations'] = 100
            if 'miss_threshold' not in optim_config:
                optim_config['miss_threshold'] = 0
        # 加载数据，并过滤掉分数过小的框
        pred_1 = load_pickle(path_pred_1)
        pred_2 = load_pickle(path_pred_2)
        gt = load_pickle(path_gt)
        pred_1 = filter_scores(pred_1)
        pred_2 = filter_scores(pred_2)
        # 将输入数据转换成me的格式
        if self.fusion_type == 'detection':
            pred_1 = convert_format_me(pred_1, self.class_id_list)
            pred_2 = convert_format_me(pred_2, self.class_id_list)
            gt = convert_format_me(gt, self.class_id_list)
        if self.fusion_type == 'hybrid':
            pred_1 = convert_format_me(pred_1, self.class_id_list)
            gt = convert_format_me(gt, self.class_id_list)
        # 初始化优化器
        self.model_ensemble.initialize_optimizer(predictions1=pred_1, predictions2=pred_2, ground_truth=gt)
        # 执行优化操作
        self.optimal_param = self.model_ensemble.maximize(metric=optim_config['metric'],
                                                          miss_threshold=optim_config['miss_threshold'],
                                                          class_names=self.class_id_list,
                                                          optimizer=optim_config['optimizer'],
                                                          max_iterations=optim_config['max_iterations'])

        # 优化完成后在命令行显示最优参数
        self.optimal_param.dump()

    def save_param(self, output_path):
        """保存最优融合超参数
        """
        if not self.optimal_param:
            raise ValueError('最优融合超参数为None，请先执行optimization操作！')
        dump_pickle(self.optimal_param, output_path)

    def load_param(self, input_path):
        """将最优融合超参数加载到类中
        """
        self.optimal_param = load_pickle(input_path)
        self.model_ensemble.initialize_parameter(parameter=self.optimal_param)

    def fuse(self, pred_1, pred_2):
        """融合单张图片的预测结果
        """
        if not self.optimal_param:
            raise ValueError('最优融合超参数为None，无法做结果融合！')
        pred_1 = convert_format_me([pred_1], self.class_id_list)[0]
        pred_2 = convert_format_me([pred_2], self.class_id_list)[0]
        return self.model_ensemble.fuse(pred_1, pred_2)
    

