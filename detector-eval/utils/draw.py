import cv2
import numpy as np

def draw_instances(image, boxes, labels=None, color=(255,0,0), disp_margin=5, thickness=1):
    """
    Draw detection results on the image
    param:
        image: 3-channel image
        boxes: a list boxes coordinates: [[xmin, ymin, xmax, ymax], ...]
        labels: a list of strings that correspond to each element in param 'boxes', when set to 'None', no labels would be draw
        color: color you want to draw 'boxes' and 'labels'
        disp_margin: a margin that put to 'boxes' in order to better show small objects encircled by 'boxes'
        thickness: thickness of fonts and lines
    return:
        image with boxes and labels (if any) drawed on it. the input image would not be changed
    """
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3, "The input image is a not rgb image"

    image_show = image.copy()  # draw on a image clone so that the input image keeps intact

    if labels is None:
        labels = [''] * len(boxes)

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        # add a margin to display small boxes better
        pt1 = (x1-disp_margin,y1-disp_margin)
        pt2 = (x2+disp_margin,y2+disp_margin)
        cv2.rectangle(image_show,pt1=pt1, pt2=pt2,color=color,thickness=thickness)
        # dynamically show 
        pt = list(pt1)
        if pt1[0] < 5: pt[0] = pt2[0]
        if pt1[1] < 5: pt[1] = pt2[1]
        pt = tuple(pt)

        cv2.putText(image_show, text=label, org=pt,
                    fontFace=1, fontScale=1, thickness=thickness, color=color)
    return image_show