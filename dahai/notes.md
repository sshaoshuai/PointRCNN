# Notes for modify PointRCNN

os.path.basename    返回path最后的文件名, 配合split使用效果更佳

和nn.Module不同，调用tensor.cuda()只是返回这个tensor对象在GPU内存上的拷贝，而不会对自身进行改变。
因此必须对tensor进行重新赋值，即tensor=tensor.cuda().

分析dataset数据的准备工作:

kittiRCNN:root_dir, npoints=16384, split='train', classes='Car', mode='TRAIN', random_select=True,
                 logger=None, rcnn_training_roi_dir=None, rcnn_training_feature_dir=None, rcnn_eval_roi_dir=None,
                 rcnn_eval_feature_dir=None, gt_database_dir=None
                
                
不同点在于:
    1. fat需要360°输入
    2. 而且对于目标任务 是多个种类都需要进行训练
    
    
修改策略:
    1. 改输入结构


数据说明
sample_annotation refers to any bounding box defining the position of an object seen in a sample. All location data is given with respect to the global coordinate system. Let's examine an example from our sample above.
数据都是在global coordinate中表示


准备工作:

ln -sf /home/dahai/kaggle/PointRCNN/nuscenes-devkit/lyft_dataset_sdk /home/dahai/anaconda3/envs/point36/lib/python3.6/site-packages/lyft_dataset_sdk
    
Python Console:

from datetime import datetime
from functools import partial
import glob
from multiprocessing import Pool

# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.
import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt

import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm, tqdm_notebook
import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

## 关于变量或者名称有用的备注

#### In Box 
Box表示方法

```
repr_str = (
            "label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], "
            "rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, "
            "vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}"
        )
```

+ wlh_factor: Multiply width, length, height by a factor to scale the box.
+ func corners: Returns the bounding box corners. 带有朝向信息(前四个点facing forward)

##### Lyft_dataset

+ func get_sample_data: Returns the data path as well as all annotations related to that sample_data.
        The boxes are transformed into the current sensor's coordinate frame.

### PointRCNN

get_valid_flag: 判断是否在图像中,并筛选

```
pts_rect = pts_rect[pts_valid_flag][:, 0:3]
```

函数用于筛选类别

```
gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
# obj_list中是物体的标记信息
def objs_to_boxes3d(obj_list):
    boxes3d = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
    for k, obj in enumerate(obj_list):
        boxes3d[k, 0:3], boxes3d[k, 3], boxes3d[k, 4], boxes3d[k, 5], boxes3d[k, 6] \
            = obj.pos, obj.h, obj.w, obj.l, obj.ry
    return boxes3d
```