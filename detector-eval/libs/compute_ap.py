import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def get_ap(evaluator, class_names, iou_threshold):
    config = {
        'iou_threshold': iou_threshold,
        'class_names': class_names
    }
    res = evaluator.eval_instance_level(config)
    ap = '{:.1f}'.format(res['ap']*100)
    return ap

def get_rp(evaluator, class_names, iou_threshold):
    config = {
        'iou_threshold': iou_threshold,
        'score_threshold': .5,
        'class_names': class_names
    }
    res = evaluator.eval_image_level(config)
    rp = '{:.1f}|{:.1f}'.format(res['recall']*100, res['precision']*100)
    return rp

def make_instance_eval_table1(evaluator, iou_threshold):
    all_classes = evaluator.class_names
    
    table_dict = dict()

    for clsname in all_classes:
        table_dict[clsname] = [get_ap(evaluator, [clsname], iou_threshold=iou_threshold)]
    table_dict['all'] = [get_ap(evaluator, evaluator.class_names, iou_threshold=iou_threshold)]

    df = pd.DataFrame(table_dict)

    return df

def make_instance_eval_table2(evaluator, iou_threshold):
    table_dict = dict()

    table_dict['Structural'] = [get_ap(evaluator, ['lianjiao', 'shakong'], iou_threshold=iou_threshold)]
    table_dict['Non-Structure'] = [get_ap(evaluator, ['huashang', 'yanghua', 'heidian', 'wuzi', 'yiwu'], iou_threshold=iou_threshold)]

    df = pd.DataFrame(table_dict)

    return df

def make_image_eval_table1(evaluator, iou_threshold):
    all_classes = evaluator.class_names
    
    table_dict = dict()


    for clsname in all_classes:
        table_dict[clsname] = [get_rp(evaluator, [clsname],iou_threshold=iou_threshold)]
    table_dict['all'] = [get_rp(evaluator, evaluator.class_names,iou_threshold=iou_threshold)]

    for clsname in all_classes:
        table_dict[clsname].append(get_rp(evaluator, evaluator.class_names, iou_threshold=iou_threshold))
    table_dict['all'].append(get_rp(evaluator, evaluator.class_names, iou_threshold=iou_threshold))

    df = pd.DataFrame(table_dict)

    return df

def make_image_eval_table2(evaluator, iou_threshold):
    table_dict = dict()

    table_dict['Structural'] = [get_rp(evaluator, ['lianjiao', 'shakong'], iou_threshold=iou_threshold)]
    table_dict['Non-Structure'] = [get_rp(evaluator, ['huashang', 'yanghua', 'heidian', 'wuzi', 'yiwu'], iou_threshold=iou_threshold)]

    df = pd.DataFrame(table_dict)

    return df

def draw_pr(evaluator, iou_threshold, class_names):
    config = {
        'iou_threshold': iou_threshold,
        'class_names': class_names
    }
    res = evaluator.eval_instance_level(config)
    recalls = res['recalls']
    precisions = res['precisions']

    fig = plt.figure(1, figsize=(10,8))
    ax = fig.gca()   # Get Current Axis
    ax.cla()         # clear existing plot
    plt.plot(recalls, precisions)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.grid()
    plt.title('AP: {:.2f}%'.format(res['ap']*100))
    return fig

def draw_pr_curves(evaluator, iou_threshold, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig = draw_pr(evaluator, iou_threshold, evaluator.class_names)
    fig.savefig(os.path.join(save_dir, 'ng-pr-curve.png'))

    for class_name in evaluator.class_names:
        fig = draw_pr(evaluator, iou_threshold, [class_name])
        fig.savefig(os.path.join(save_dir, '{}-pr-curve.png'.format(class_name)))


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from evaluator import Evaluator4Classifier, Evaluator4Detector

    product = '3L-new'
    experiment = 'exp9.bigshakong'

    gt_det_json = '/mnt/mt4_tmp/zhaimenghua/zhaimenghua/experiments/results/{}/{}/manual/det.json'.format(product, experiment)
    gt_cls_json = '/mnt/mt4_tmp/zhaimenghua/zhaimenghua/experiments/results/{}/{}/manual/cls.json'.format(product, experiment)
    vi_det_json = '/mnt/mt4_tmp/zhaimenghua/zhaimenghua/experiments/results/{}/{}/vi/det.json'.format(product, experiment)
    vi_cls_json = '/mnt/mt4_tmp/zhaimenghua/zhaimenghua/experiments/results/{}/{}/vi/cls.json'.format(product, experiment)
    gth_json = '/mnt/mt4_tmp/zhaimenghua/zhaimenghua/experiments/results/{}/gths/manual.json'.format(product)
    vigt_json = '/mnt/mt4_tmp/zhaimenghua/zhaimenghua/experiments/results/{}/gths/vi.json'.format(product)
    vidt_json = '/mnt/mt4_tmp/zhaimenghua/zhaimenghua/experiments/results/{}/gths/vi_as_det.json'.format(product)


    # iou_threshold=1e-8
    # ev = Evaluator4Detector(
    #     det_json = gt_det_json,
    #     gth_json = gth_json
    # )
    # ev = Evaluator4Detector(
    #     det_json = vi_det_json,
    #     gth_json = vigt_json
    # )


    iou_threshold=.999999
    ev = Evaluator4Classifier(
        det_json = gt_cls_json,
        gth_json = gth_json
    )
    # ev = Evaluator4Classifier(
    #     det_json = vi_cls_json,
    #     gth_json = vigt_json
    # )

    draw_pr_curves(ev, iou_threshold, '/tmp/test/')