# Model Ensemble (ME) API使用说明

该API目前只支持多个分类网络的Model Ensemble和两个检测网络（或是分类检测的混合网络）的融合。

## 配置环境

ME API应该作为一个子模块存在于你的项目中。在你的项目文件夹中添加ME API子模块的操作如下：

``` shell
git submodule add [GIT_URL.git] me_api
git submodule update --init --recursive
```



本API所有用到的安装包都在requirements.txt里。如果你有PyPI (pip)，运行以下命令安装依赖：

``` shell
pip install -r requirements.txt
```



## 使用说明

### 融合类型

目前我们支持以下几个类型的模型融合（2模型融合）：

- **multi-class**: Multi-Class + Multi-Class = Multi-Class
- **multi-label**: Multi-Label + Multi-Label = Multi-Label
- **detection**: Detection + Detection = Detection
- **hybrid**: Detection + Multi-Label = Detection（**注意：模型1是detection模型，模型2是multi-label模型，次序不可搞混**）

调用该API之前，请先弄清楚自己的实验属于以上哪些融合类型。融合类型确认以后，用以下代码import相应融合类型所需的类声明：

``` python
from me_api import import_me_classes
ModelEnsemble, ParameterClass = import_me_classes(mode='hybrid')  # 调用hybrid融合
```

函数import_me_classes()的参数mode可以取以上列出的几种融合类型。



可以直接运行demo.py查看融合效果：

``` shell
python3 demo.py
```

我们现在逐行解析demo.py

### 数据加载

``` python
import sys
sys.path.append('model_ensemble/')

# import necessary classes for detection model ensemble
# all possible mode: ["detection", "multi-class", "multi-label"]
from me_api import import_me_classes
ModelEnsemble, ParameterClass = import_me_classes(mode='detection')

# Prepare data
import pickle
with open('assets/data.pkl', 'rb') as fid:
    dat = pickle.load(fid)
predictions1_val = dat['val_preds1']
...
```

**在调用API前，首先要将路径"me_api/model_ensemble/"加入到你的python系统环境中。**

这部分代码还干了两件事：

1. import必要的类对象：ModelEnsemble和ParameterClass；**这两个后面会用到的类是通过me_api.import_me_classes()函数返回的，该函数需要用于预先设定好融合模型的类别**
2. 加载预先存储好的数据（推断结果和标注都存储在"assets/data.pkl"）

加载进来的数据包括validation set的推断结果和标注，test set的推断结果和标注。其中：

- predictions1_val：模型1在validation set上的推断结果
- predictions2_val：模型2在validation set上的推断结果
- ground_truth_val：validation set上的标注
- predictions1_test：模型1在test set上的推断结果
- predictions2_test：模型2在test set上的推断结果

对不同模型来说，模型的推断结果的数据结果也有所不同，根据类型可以划分成以下两类：

#### 1. 检测网络推断结果的数据结构：

对于检测网络来说，网络的推断结果是一个数组结构：

``` python
[
    [],
    [([0.1, 0.3, 0.5, 0.05, 0.05], [150, 11, 200, 60]), ([0.75, 0.12, 0.01, 0.02, 0.1], [55, 110, 77, 125])]
]
```

该数组有两个元素，分别代表模型在两张图片上的检测结果。该模型在第一张图片上的检测结果为空，所以第一个元素是个空列表。

该模型在第二张图片上检出两个框，其中第一个检测框存储为一个tuple：

``` python
([0.1, 0.3, 0.5, 0.05, 0.05], [150, 11, 200, 60])
```

左边的数组是该检测框对应的distribution，其中第一个元素0.1是背景类的概率，后面四个元素为前景类的概率。

右边的数组是检测框的位置信息：[Xmin, Ymin, Xmax, Ymax]。

标注也用同样的数据结构存储，只不过标注的distribution有以下两个性质：

1. 背景类概率为0（即distribution数组的第一个元素为0）
2. 其它元素有且只有一个元素为1，其余元素都为0

predictions1_val, predictions2_val, 和ground_truth_val里的元素应一一对应，即第i个元素应代表第i张图片的检测结果和标注。

#### 2. 分类网络推断结果的数据结构：

对于分类网络来说，网络的推断结果也是一个数组结构：

``` python
[
    [0.1, 0.3, 0.5, 0.05, 0.05],
    [0.75, 0.12, 0.01, 0.02, 0.1]
]
```

该数组有两个元素，分别代表模型在两张图片的预测结果。每张图只有一个长度为类别总个数的向量（称为分布向量），该向量代表了每个类的得分/概率。如果是多分类任务，则该向量所有元素之和为一，如果是多标签，则该向量所有元素的值都大于0小于1。

标注也用同样的数据结构存储。



predictions1_val, predictions2_val, 和ground_truth_val里的元素应一一对应，即第i个元素应代表第i张图片的检测结果和标注。



### ME超参搜索

下面我们来看用validation set寻找ME超参的代码：

``` python
me_for_optimization = ModelEnsemble()
# feed validation data to the optimizer
me_for_optimization.initialize_optimizer(
    predictions1 = predictions1_val,
    predictions2 = predictions2_val,
    ground_truth = ground_truth_val
)
# set metric to optimize, and the optimizer type
optimal_param = me_for_optimization.maximize(metric='mAP', optimizer='bayesian')
optimal_param.dump()  # display parameters on terminal
optimal_param.pickle(pickle_path='optim_param.pkl')
```

#### 加载验证数据

首先我们初始化一个ModelEnsemble类实例：

``` python
me_for_optimization = ModelEnsemble()
```

然后把之前加载的validation set的预测和标注数据喂进该实例：

``` python
me_for_optimization.initialize_optimizer(
    predictions1 = predictions1_val,
    predictions2 = predictions2_val,
    ground_truth = ground_truth_val
)
```

如果想融合多个网络模型（目前只支持分类网络多模型融合），加载数据依次未给predictions1, predictions2, predictions3, ...即可：

```python
me_for_optimization.initialize_optimizer(
    predictions1 = predictions1_val,
    predictions2 = predictions2_val,
    predictions3 = predictions3_val,
    ...
    ground_truth = ground_truth_val
)
```

数据初始化完成后，就可以运行超参优化过程：

``` python
optimal_param = me_for_optimization.maximize(metric='mAP', optimizer='bayesian')
```

maximize()函数返回一个ParameterClass类实例，该实例管理ME参数的存储与加载，之后我们会介绍到。

#### Metric的选择

我们的API支持多种metric的优化，其中包括：

- '**mAP**'：所有类别（对检测模型来说是所有前景类别）的平均AP
- '**ngAP**'：所有前景类别合并成NG类的AP值（前景类class_id>0的所有类别）；'ngAP'强调对缺陷的检出性能而非分类性能；**'ngAP'只能用于评估检测模型**。
- '**classAP**'：指定合并类别的AP值；需要在maximize()函数中设置参数```class_names```的值，该参数读取一个类别的列表，类别名称默认格式是'classX'，其中'X'是类别标签（一个从0开始的整数）
- '**classmAP**'：指定制定类别的mAP值；需要在maximize()函数中设置参数```class_names```的值，该参数读取一个类别的列表，类别名称默认格式是'classX'，其中'X'是类别标签（一个从0开始的整数）
- '**prec@rec**'：某召回率下的精度 (precition at recall)；需要设置参数```class_names```的值和参数```recall_threshold```的值（该值的范围在0到1之间）；对detection任务来说，某些recall可能达不到，所以我们在计算检测模型的'prec@rec'时会先把检测模型的输出退化成图像级别的multi-label输出（每个类别的分数取所有检测框该类别的概率最大值），再计算'prec@rec'。

maximize()函数还有一个参数是```iou_threshold```，在检测模型融合的评估中，```iou_threshold```的默认值是0.1。

下面我们来看一个复杂写的maximize()方法的调用，该代码用bayesian optimizer最优化类别2和类别5（在检测的例子中，假设包括背景类一共有6个类别，其中'class0'是前景类别）合并后的'prec@rec=0.4'：

``` python
optimal_param = me_for_optimization.maximize(  # A complicated metric optimization
     metric='prec@rec', optimizer='bayesian',
     class_names=['class5', 'class2'], iou_threshold=0.1, recall_threshold=0.4
)
```



除了使用现成的metrics，你也可以定义自己的metric，只需要定义一个callback函数如下：

``` python
def my_metric(predictions, ground_truth):
    metric = 0
    for preds, gths in zip(predictions, ground_truth):
        num_preds = len([x for x in preds if x[0][0]<0.5])
        num_gths  = len(gths)
        metric -= abs(num_preds-num_gths)
    return metric
```

自定一的metric函数输入两个参数，输出一个浮点数。

其中函数的两个参数，predictions和ground_truth，分别是数据集的预测和结果和标注，结构和predictions1_val与ground_truth_val的结构一致。my_metric()函数计算了数据集上NG预测框个数与标注框个数之间的差异。

调用优化函数时，把my_metric()函数当做metric参数的输入即可：

```python
optimal_param = me_for_optimization.maximize(metric=my_metric, optimizer='bayesian')
```

#### 超参优化器的选择

optimizer有以下几个可选项：

``` python
['bayesian', 'passive', 'random', 'genetic']
```

我们推荐'bayesian'和'genetic'。'bayesian'即Bayesian Optimizer（贝叶斯优化器），比较慢但是性能比其他几个都好；'genetic'是遗传算法优化器，优点是速度快，性能在'random'和'bayesian'之间。

参数优化是个耗时较长的过程，所以优化好后，我们把最优的参数保存在pickle文件中：

```python
optimal_param.pickle(pickle_path='optim_param.pkl')
```



### 模型融合

有了ME参数以后，我们的API能对单张图片的预测结果做模型融合。但首先，我们从预先存储好的超参文件加载超参：

``` python
me = ModelEnsemble()  # 声明一个新ModelEnsemble实例（当然也可以用上面的me_for_optimization）
param = ParameterClass.from_pickle(pickle_path='optim_param.pkl')  # 从文件加载ME超参
me.initialize_parameter(parameter=param)  # 初始化ModelEnsemble实例的超参
```

对单张图片做融合的代码如下：

``` python
for pred1, pred2 in zip(predictions1_test, predictions2_test):
    pred_fused = me.fuse(pred1, pred2)
```

**注意pred1和pred2的输入次序不能搞混！**

如果涉及多个模型，则需要输入更多参数，可以向下面这样：

```python
pred_fused = me.fuse(pred1, pred2, pred3, ...)
```

me.fuse()函数的输出和输入有相同的数据结构，相关定义见上文。



### 数据评估

ME API顺带还提供简单的评估功能。要使用该功能也很简单，首先要设置好融合参数和评估数据集：

``` python
me_for_eval = ModelEnsemble()
# 加载、设置参数
param = ParameterClass.from_pickle(pickle_path='optim_param.pkl')
me_for_eval.initialize_parameter(parameter=param)
# 设置评估数据（假装我们有predictions1/2，和ground_truth）
me_for_eval.initialize_evaluator(
    predictions1 = predictions1,
    predictions2 = predictions2,
    ground_truth = ground_truth
)
# 评估 'mAP'
me_for_eval.eval(metric='mAP')
```

方法initialize_evaluator()也支持多模型融合，具体参数设置和方法initialize_optimizer()一样。

和上面提到过的优化方法maximize()一样，评估方法eval()的metric也可以有其它选择，具体参见maximize()的输入参数说明。举一个复杂点的评估例子：

``` python
# 评估recall=0.9时，第二和第五个前景类别合并后的precision是多少
me_for_eval.eval(
    metric='prec@rec',
    class_names=['class2', 'class5'],
    recall_threshold=0.9
)
```

评估结果包含两个单模型在该数据集下，融合前的metric分别是多少；还有它们融合之后的模型在指标是多少。结果打印在屏幕上。

