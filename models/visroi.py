import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def visual_roi(targets, feat):
    path = '/home/hanzhixiong/hanzhixiong/SeqNet-master/data/PRW/frames/' + targets[0]['img_name']
    # Load the original image
    ori_image = cv2.imread(path)
    boxes = targets[0]['boxes'].cpu().numpy()
    for idx in range(boxes.shape[0]):
        if targets[0]['labels'][idx] == 5555:
            continue
        image = deepcopy(ori_image)
        box = boxes[idx]
        # Select the ROI and create a blank image with the same size
        roi = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        cv2.imwrite('/home/hanzhixiong/hanzhixiong/SeqNet-master/logs/vis_roi_feat4/' + targets[0]['img_name'][:-4] + '_' + str(idx)+ '_ori' +'.png', roi)
        h, w, _ = roi.shape
        values = feat[idx].permute(1, 2, 0).cpu().numpy()
        values = cv2.resize(values, (w, h))
        values = values.mean(axis=-1)

        normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))

        # Use the jet color map to map the normalized values to colors
        colors = plt.cm.jet(normalized_values)

        # Convert the colors to integers in the range 0-255
        colors = (colors * 255).astype(np.uint8)
        colors_3ch = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR)
        colors_3ch = cv2.cvtColor(colors_3ch, cv2.COLOR_BGR2RGB)
        # Overlay the heat map on top of the original image using alpha blending
        alpha = 0.3
        
        cv2.addWeighted(colors_3ch, alpha, roi, 1 - alpha, 0, roi)

        # Display the resulting image with the heat map overlaid on top
        cv2.imwrite('/home/hanzhixiong/hanzhixiong/SeqNet-master/logs/vis_roi_feat4/' + targets[0]['img_name'][:-4] +'_'+ str(idx) +'.png', roi)
    cv2.waitKey(0)
