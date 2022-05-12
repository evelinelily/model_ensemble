"""MatrixEnsemble接口类"""
from me_methods import get_me_classes
from utils.data_processing import load_pickle, load_json, dump_json, filter_scores, convert_format_me, convert_format_det


class MatrixEnsemble(object):

    def __init__(self, fusion_type, class_id_list=None):
        self.fusion_type = fusion_type
        self.class_id_list = class_id_list
        # 多类别和多标签分类都统一到multilabel上
        if fusion_type == 'classification':
            self.ModelEnsemble, self.ParameterClass = get_me_classes('multi-label')
        else:
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
        # 将输入数据转换成me的格式
        if self.fusion_type == 'detection':
            pred_1 = filter_scores(pred_1)
            pred_2 = filter_scores(pred_2)
            pred_1 = convert_format_me(pred_1, self.class_id_list)
            pred_2 = convert_format_me(pred_2, self.class_id_list)
            gt = convert_format_me(gt, self.class_id_list)
        if self.fusion_type == 'hybrid':
            pred_1 = filter_scores(pred_1)
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
        param = dict()
        if self.fusion_type == 'detection':
            param['iou_threshold'] = self.optimal_param.iou_threshold
            param['roi_weight'] = self.optimal_param.roi_weight
            param['lonely_fg_weight1'] = self.optimal_param.lonely_fg_weight1
            param['lonely_fg_weight2'] = self.optimal_param.lonely_fg_weight2
            param['dist_weights'] = self.optimal_param.dist_weights
            param['num_classes'] = self.optimal_param.num_classes
        else:  # classification、hybrid都当做classification处理
            param['dist_weights'] = self.optimal_param.dist_weights
            param['num_classes'] = self.optimal_param.num_classes
            param['num_models'] = self.optimal_param.num_models
        dump_json(param, output_path)

    def load_param(self, param_path):
        """将最优融合超参数加载到类中
        """
        param = load_json(param_path)
        self.optimal_param = self.ParameterClass()
        if self.fusion_type == 'detection':
            self.optimal_param.iou_threshold = param['iou_threshold']
            self.optimal_param.roi_weight = param['roi_weight']
            self.optimal_param.lonely_fg_weight1 = param['lonely_fg_weight1']
            self.optimal_param.lonely_fg_weight2 = param['lonely_fg_weight2']
            self.optimal_param.dist_weights = param['dist_weights']
            self.optimal_param.num_classes = param['num_classes']
        else:  # classification、hybrid都当做classification处理
            self.optimal_param.dist_weights = param['dist_weights']
            self.optimal_param.num_classes = param['num_classes']
            self.optimal_param.num_models = param['num_models']

        self.model_ensemble.initialize_parameter(parameter=self.optimal_param)

    def fuse(self, pred_1, pred_2):
        """融合单张图片的预测结果。注意：输入和输出数据均为正常的目标检测格式，即每个实例为[x1, y1, x2, y2, score, class_id]。
        """
        if not self.optimal_param:
            raise ValueError('最优融合超参数为None，无法做结果融合！')
        if self.fusion_type == 'detection':
            pred_1 = convert_format_me([pred_1], self.class_id_list)[0]
            pred_2 = convert_format_me([pred_2], self.class_id_list)[0]
            pred_fused = self.model_ensemble.fuse(pred_1, pred_2)
            pred_fused = convert_format_det([pred_fused])[0]
        elif self.fusion_type == 'classification':
            pred_fused = self.model_ensemble.fuse(pred_1, pred_2)
        else:  # hybrid
            pass
        return pred_fused
