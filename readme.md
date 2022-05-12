# 1 功能描述

MatrixEnsemble是一个model-free的结果融合方法，支持以下几种模式的融合：

1. 分类 + 分类
2. 目标检测 + 目标检测
3. 分类 + 目标检测

# 2 接口类说明

```python
class MatrixEnsemble(object):
    def __init__(self, fusion_type, class_id_list=None):
        XXX

    def optimization(self, path_pred_1, path_pred_2, path_gt, optim_config=None):
        XXX

    def save_param(self, output_path):
        XXX

    def load_param(self, input_path):
        XXX

    def fuse(self, pred_1, pred_2):
        XXX
        return pred_fused
```

方法及参数描述：

### 2.1 \_\_init\_\_方法

* 功能
  * 指定融合模式
* 入参
  * fusion_type: 融合模式，str型，取值范围{ 'classification', 'detection','hybrid'}，分别对应：“分类+分类”，“检测+检测”，“分类+检测”。
  * class_id_list: 模型覆盖的总的类别列表，list型，主要用于检测模型中防止类别缺失的情况出现，如总类别为[1, 2, 3, 4, 5]，但模型1中只出现了[1, 3, 4, 5]，模型2中只出现了[2, 3, 4]。该参数“分类+分类”的融合模式可以不指定，其他模式必须要指定。

* 返回值
  * 无

### 2.2 optimization方法

* 功能
  * 执行优化操作，获取最优融合超参数。注意，若为hybrid模式的融合，则第1个模型必须为检测模型，第2个模型必须为分类模型，且gt为检测格式的gt。

* 入参
  * path_pred_1: 用于做融合的第1个模型的结果数据路径，str型，目前只支持pkl，具体参见第3节说明。
  * path_pred_2: 用于做融合的第2个模型的结果数据路径，str型，目前只支持pkl，具体参见第3节说明。
  * path_gt: 用于做融合的真值结果数据路径，str型，目前只支持pkl，具体参见第4节说明。
  * optim_config: 做优化需要的参数配置，dict型，此字典包含如下几个具体参数，注意以下参数均为可选参数，可以不指定。
    * metric: 评价指标，str型，取值范围{'mAP', 'ngAP', 'prec@rec', 'rec@prec'}，默认为'mAP'。
    * optimizer: 优化方法，str型，取值范围{'bayesian', 'passive', 'random', 'genetic'}，默认为'bayesian'。
    * max_iterations: 最大迭代次数，int型，默认值为100。
    * miss_threshold: 最小漏报率要求，float型，默认值为0。

* 返回值
  * 无

### 2.3 save_param方法

* 功能
  * 保存最优融合超参数。
* 入参
  * output_path: 最优超参数路径，str型。

* 返回值
  * 无

### 2.4 load_param方法

* 功能
  * 加载最优融合超参数。

* 入参
  * input_path: 最优超参数路径，str型。

* 返回值
  * 无

### 2.5 fuse方法

* 功能
  * 采用最优超参数对两条预测结果进行融合。

* 入参
  * pred_1: 第1个模型的一条预测结果， list型，具体参见第3节说明。
  * pred_2: 第2个模型的一条预测结果， list型，具体参见第3节说明。

* 返回值
  * pred_fused: 融合后的结果

# 3 **数据格式说明**

path_pred_1, path_pred_2, path_gt均为pkl文件，其中存储的是目标检测结果或者分类结果，它是一个将python中list型的数据进行二进制序列化后的文件。

### 3.1 检测模型的结果示例

```python
[
  [[30, 234, 145, 370, 0.7212, 3]],
  [],
  [[10, 67, 89, 188, 0.8315, 2], [300, 230, 388, 315, 0.5634, 4]]
]
```

以上结果为3张图片的检测结果，第1张图片有一个实例，第2张图片没有，第3张图片有2个实例。各个实例的数据格式为：[x1, y1, x2, y2, score, class_id]，对应类型分别为[int, int, int, int, float, int]。gt中的score统一设为1。

### 3.2 分类模型的结果示例

```python
[
  [0.6425, 0.0000, 0.0000, 0.0004, 0.0012],
  [0.0000, 0.0001, 0.0089, 0.8798, 0.1137]
]
```

分类结果用one-hot形式表示。以上结果为2张图片在一个5分类模型上的分类结果，每张图片的结果为一个长度为类别总数的list。注意multiclass中同一张图片的预测结果之和为1， multilabel中同一张图片的预测结果之和不一定为1。

# 4 使用示例

### 4.1 离线超参优化（类似训练）

```python
# 参数设置
path_pred_1 = './liyu_data/ori_input/yolo/val/pred.pkl'
path_pred_2 = './liyu_data/ori_input/ssd/val/pred.pkl'
path_gt = './liyu_data/ori_input/yolo/val/gt.pkl'
optim_config = {'metric': 'mAP', 'optimizer': 'bayesian', 'max_iterations': 100}
# 执行优化
me = MatrixEnsemble(fusion_type='detection', class_id_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
me.optimization(path_pred_1, path_pred_2, path_gt, optim_config)
me.save_param('./output/param.pkl')
```

### 4.2 在线推理（类似测试）

```python
# 参数设置
path_ts_1 = './liyu_data/ori_input/yolo/test/pred.pkl'
path_ts_2 = './liyu_data/ori_input/ssd/test/pred.pkl'
path_ts_gt = './liyu_data/ori_input/yolo/test/gt.pkl'
# 执行推理
me = MatrixEnsemble(fusion_type='detection', class_id_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
me.load_param('./output/param.pkl')
predictions_1 = load_pickle(path_ts_1)
predictions_2 = load_pickle(path_ts_2)
predictions_fused = []
for pred_1, pred_2 in zip(predictions_1, predictions_2):
  pred_fused = me.fuse(pred_1, pred_2)
  predictions_fused.append(pred_fused)
```