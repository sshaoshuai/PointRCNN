#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lyft_dataset.py.py    
@Contact :   alfredo2019love@outlook.com
@License :   (C)Copyright 2019-2020, DRL_Lab-Cheng-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
10/1/19 7:22 PM   Cheng      1.0         init
'''

# This file is to create a dataloader and transfer the train and gt to network
import numpy as np
import torch
import torch.utils.data as torch_data
import pandas as pd

import random

from lib.lyft_config import cfg
import lib.utils.kitti_utils as kitti_utils
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
from lib.utils.data_processing import convert_pointrcnn_coners, filtrate_objects, convert_pointrcnn_boxes


class LyftDataloader(torch_data.Dataset):
    def __init__(self, level5data, shuffle=False, mode='TRAIN', npoints=65536,
                 classes=None):
        if classes is None:
            classes = ['car', 'bus', 'bicycle', 'emergency_vehicle', 'motorcycle', 'other_vehicle',
                       'pedestrian', 'truck']
            # classes = ['car', 'bus', 'animal', 'bicycle', 'emergency_vehicle', 'motorcycle', 'other_vehicle',
            #            'pedestrian', 'truck']
        self.classes = classes
        self.num_class = self.classes.__len__()

        self.lyft_dataset = level5data

        self.sample_list = level5data.sample.copy()
        self.random_select = shuffle
        self.mode = mode
        self.npoints = npoints
        if shuffle:
            random.shuffle(self.sample_list)

    # if classes == 'Car':
    #     self.classes = ('Background', 'Car')
    #     aug_scene_root_dir = os.path.join(root_dir, 'KITTI', 'aug_scene')
    # elif classes == 'People':
    #     self.classes = ('Background', 'Pedestrian', 'Cyclist')
    # elif classes == 'Pedestrian':
    #     self.classes = ('Background', 'Pedestrian')
    #     aug_scene_root_dir = os.path.join(root_dir, 'KITTI', 'aug_scene_ped')
    # elif classes == 'Cyclist':
    #     self.classes = ('Background', 'Cyclist')
    #     aug_scene_root_dir = os.path.join(root_dir, 'KITTI', 'aug_scene_cyclist')
    # else:
    #     assert False, "Invalid classes: %s" % classes

    def __len__(self):
        return self.sample_list.__len__()

    def __getitem__(self, index):
        # return lidar file path, boxes in sensor coordinate
        sample_id = self.sample_list[index]['token']
        if sample_id == ''
        data_path, boxes, _ = self.lyft_dataset.get_sample_data(self.sample_list[index]['data']['LIDAR_TOP'])
        lidar_pc = LidarPointCloud.from_file(data_path).points.T

        pts_rect = lidar_pc[:, :3]
        pts_intensity = lidar_pc[:, 3]

        # generate inputs
        if self.mode == 'TRAIN' or self.random_select:
            # print('nums of pts_rect: ', len(pts_rect))
            if self.npoints < len(pts_rect):
                pts_depth = pts_rect[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=False)

                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(pts_rect), dtype=np.int32)
                if self.npoints > len(pts_rect):
                    extra_choice = np.random.choice(choice, self.npoints - len(pts_rect),
                                                    replace=False)  # origin: Flase
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            ret_pts_rect = pts_rect[choice, :]
            ret_pts_intensity = pts_intensity[choice] / 100.0 - 0.5  # translate intensity to [-0.5, 0.5]
        else:
            ret_pts_rect = pts_rect
            ret_pts_intensity = pts_intensity - 0.5

        sample_info = {'sample_id': sample_id, 'random_select': self.random_select}

        pts_features = [ret_pts_intensity.reshape(-1, 1)]
        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]

        if self.mode == 'TEST':
            if cfg.RPN.USE_INTENSITY:
                pts_input = np.concatenate((ret_pts_rect, ret_pts_features), axis=1)  # (N, C)
            else:
                pts_input = ret_pts_rect
            sample_info['pts_input'] = pts_input
            sample_info['pts_rect'] = ret_pts_rect
            sample_info['pts_features'] = ret_pts_features
            return sample_info
        # ---------Start to handle gt_box and labels

        gt_obj_list = filtrate_objects(boxes)
        # if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN' and gt_aug_flag:
        #     gt_obj_list.extend(extra_gt_obj_list)
        gt_boxes3d_like = convert_pointrcnn_boxes(gt_obj_list)
        # 未经过数据增强方法:防止第一阶段出错,也不是使用反射强度信息

        # prepare input
        if cfg.RPN.USE_INTENSITY:
            pts_input = np.concatenate((ret_pts_rect, ret_pts_features), axis=1)  # (N, C)
            # ((aug_pts_rect, ret_pts_features), axis=1)  # (N, C)
        else:
            pts_input = pts_rect  # aug_pts_rect

        # 此处的RPN FIXED提前返回与之前不同,返回的是增强过之后的数据,暂且不用
        # if cfg.RPN.FIXED:
        #     sample_info['pts_input'] = pts_input
        #     sample_info['pts_rect'] = aug_pts_rect
        #     sample_info['pts_features'] = ret_pts_features
        #     sample_info['gt_boxes3d'] = aug_gt_boxes3d
        #     return sample_info

        # generate training labels
        rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(ret_pts_rect, gt_obj_list)
        sample_info['pts_input'] = ret_pts_rect
        sample_info['pts_rect'] = ret_pts_rect
        sample_info['pts_features'] = ret_pts_features
        sample_info['rpn_cls_label'] = rpn_cls_label
        sample_info['rpn_reg_label'] = rpn_reg_label
        sample_info['gt_boxes3d'] = gt_boxes3d_like
        return sample_info

        # rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(lidar_pc, lidar_pc)

        # sample_info['pts_input'] = lidar_pc
        # sample_info['pts_rect'] = lidar_pc
        # sample_info['pts_features'] = lidar_pc
        # sample_info['rpn_cls_label'] = rpn_cls_label
        # sample_info['rpn_reg_label'] = rpn_reg_label
        # sample_info['gt_boxes3d'] = lidar_pc

        # return self.sample_info

    def generate_rpn_training_labels(self, pts_rect, gt_boxes3d):
        """

        :param pts_rect: (npoints, 3)
        :param gt_boxes3d: PointRCNN中为(N, 7) x, y, z, h, w, l, ry 此处的ry为绕竖直轴旋转后的
        此处的输入为lyftdataset-Box对象 (N, ??)
        :return:
        """
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        reg_label = np.zeros((pts_rect.shape[0], 7), dtype=np.float32)  # dx, dy, dz, w, l, h, ry
        # The gt_corners are already rotated and transformed
        gt_corners = convert_pointrcnn_coners(gt_boxes3d)  # (N, 8, 3) N为box的数量
        # extend_gt_boxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2) # 暂时不需要这个操作
        extend_gt_boxes3d = 1.2  # 原始是扩大之后的框,现在用缩放因子代替
        # extend_gt_corners = kitti_utils.boxes3d_to_corners3d(extend_gt_boxes3d, rotate=True)
        extend_gt_corners = convert_pointrcnn_coners(gt_boxes3d, wlh_factor=extend_gt_boxes3d)

        for k in range(gt_boxes3d.__len__()):
            box_corners = gt_corners[k]
            # TODO 三角网是怎么起作用的不大清楚
            # fg_pt_flag标记在框内的点,(0~npoints, True/False)
            fg_pt_flag = kitti_utils.in_hull(pts_rect, box_corners)  # pts_rect(npoints, 3), box_corners(8, 3)
            # fg_pts_rect返回在框内的点(0~n, 3)
            fg_pts_rect = pts_rect[fg_pt_flag]
            # 标记这些点类别为1, 由fg_pt_flag标记作用
            cls_label[fg_pt_flag] = 1

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            # fg_enlarge_flag标记在增大框内的点,(0~npoints, True/False)
            fg_enlarge_flag = kitti_utils.in_hull(pts_rect, extend_box_corners)
            # 忽略标记:负样本,在增大的框内但是不在标注框内的点,对两次的标志位抑或操作
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            # 进行打标签
            cls_label[ignore_flag] = -1

            # pixel offset of object center;already the center of box
            reg_label[fg_pt_flag, 0:3] = gt_boxes3d[k].center[:3]

            # size and angle encoding
            reg_label[fg_pt_flag, 3] = gt_boxes3d[k].wlh[0]  # w
            reg_label[fg_pt_flag, 4] = gt_boxes3d[k].wlh[1]  # l
            reg_label[fg_pt_flag, 5] = gt_boxes3d[k].wlh[2]  # h
            reg_label[fg_pt_flag, 6] = gt_boxes3d[k].orientation.radians  # ry

        return cls_label, reg_label

    def collate_batch(self, batch):
        if self.mode != 'TRAIN' and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            assert batch.__len__() == 1
            return batch[0]

        batch_size = batch.__len__()
        ans_dict = {}

        for key in batch[0].keys():
            # True 进入
            if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                    (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, batch[k][key].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_boxes3d[i, :batch[i][key].__len__(), :] = batch[i][key]
                ans_dict[key] = batch_gt_boxes3d
                continue

            if isinstance(batch[0][key], np.ndarray):
                if batch_size == 1:
                    ans_dict[key] = batch[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0)

            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict


if __name__ == '__main__':
    level5data = LyftDataset(data_path='/data/sets/lyft/v1.01-train/v1.01-train',
                             json_path='/data/sets/lyft/v1.01-train/v1.01-train/v1.01-train', verbose=True)

    from tqdm import tqdm

    #
    test = LyftDataloader(level5data)
    bug_test = torch.utils.data.DataLoader(test, batch_size=18, shuffle=True,
                                           num_workers=6, pin_memory=True, collate_fn=test.collate_batch,
                                           drop_last=True)
    for sample_info in tqdm(bug_test):
        print(sample_info['sample_id'])
