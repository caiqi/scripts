import os
from mxnet import image
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
from sklearn.cluster import KMeans
import mxnet as mx
import numpy as np
import cv2
import os
from tqdm import tqdm
import json

if __name__ == '__main__':
    all_data = open("/home/caiqi/v-qcaii/research/cvpr2018/data/cityscapes/train_bbox.txt").readlines()
    all_data_new = []
    for m in all_data:
        m = m.strip().split(" ")
        path = m[0]
        annotation = [float(k) for k in m[1:]]
        annotation = np.array(annotation).reshape((-1,5)).tolist()
        for j in range(len(annotation)):
            all_data_new.append([path] + annotation[j])
    all_data = all_data_new

    classes = "back,car"
    classes = classes.split(",")
    for idx, ann in tqdm(enumerate(all_data)):
        img = cv2.imread(str(ann[0]))
        m = ann[1:]
        cv2.putText(img, "{}".format(classes[int(m[-1])]),
                    (int(m[0]), int(m[1])),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1)
        color = [int(np.random.rand() * 255), int(np.random.rand() * 255), int(np.random.rand() * 255)]
        cv2.rectangle(img, (int(m[0]), int(m[1])), (int(m[2]), int(m[3])),
                      color=color, thickness=1)
        outputu = os.path.join("cropped", classes[int(m[-1])],
                               os.path.basename(str(ann[0])).replace(".jpg", "") + "_" + classes[int(m[-1])] + "_" + str(
                                   np.random.rand()) + ".jpg")
        if not os.path.exists(os.path.dirname(outputu)):
            os.makedirs(os.path.dirname(outputu))
        cv2.imwrite(outputu, img)
