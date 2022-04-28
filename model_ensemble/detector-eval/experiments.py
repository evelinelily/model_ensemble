import os

def experiment_kangqiang_24l():
    return \
    [
        {
            'mode': 'normal',
            'exp': 'FPN',
            'det_json': '/home/zhai/Documents/MatrixTime/ML-Infrastructure/model-ensemble/assets/kangqiang/24L/FPN/test.json',
            'gt_json': '/home/zhai/Documents/MatrixTime/ML-Infrastructure/model-ensemble/assets/kangqiang/24L/gth/test.json',
            'save_dir': '/tmp/kangqiang-fusion/FPN/',
            'img_dir': '/mnt/kangqiang_data/new_data_0331/QFN-4X4-24L/',
            'save_xlsx': '/tmp/kangqiang-fusion/statistics.xlsx',
        },
        {
            'mode': 'normal',
            'exp': 'DIFF',
            'det_json': '/home/zhai/Documents/MatrixTime/ML-Infrastructure/model-ensemble/assets/kangqiang/24L/DIFF/test.json',
            'gt_json': '/home/zhai/Documents/MatrixTime/ML-Infrastructure/model-ensemble/assets/kangqiang/24L/gth/test.json',
            'save_dir': '/tmp/kangqiang-fusion/DIFF/',
            'img_dir': '/mnt/kangqiang_data/new_data_0331/QFN-4X4-24L/',
            'save_xlsx': '/tmp/kangqiang-fusion/statistics.xlsx',
        },
        {
            'mode': 'normal',
            'exp': 'Fusion',
            'det_json': '/home/zhai/Documents/MatrixTime/ML-Infrastructure/model-ensemble/assets/kangqiang/24L/fused-test.json',
            'gt_json': '/home/zhai/Documents/MatrixTime/ML-Infrastructure/model-ensemble/assets/kangqiang/24L/gth/test.json',
            'save_dir': '/tmp/kangqiang-fusion/Fusion/',
            'img_dir': '/mnt/kangqiang_data/new_data_0331/QFN-4X4-24L/',
            'save_xlsx': '/tmp/kangqiang-fusion/statistics.xlsx',
        },
    ]