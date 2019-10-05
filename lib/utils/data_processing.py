#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_processing.py    
@Contact :   alfredo2019love@outlook.com
@License :   (C)Copyright 2019-2020, DRL_Lab-Cheng-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
10/4/19 10:11 AM   Cheng      1.0         init
'''

import numpy as np


def convert_pointrcnn_boxes(box_list):
    """
    Box repr_str = (
        "label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], "
        "rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, "
        "vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}"
    )
    转换成一个7维的向量,   (x, y, z, h, w, l, ry)
    obj.pos, obj.h, obj.w, obj.l, obj.ry
    :return:
    """

    gt_boxes3d_like = np.array(
        [[box.center[0], box.center[1], box.center[2], box.wlh[0], box.wlh[1], box.wlh[2], box.orientation.radians] for
         box in box_list])

    return gt_boxes3d_like


def convert_pointrcnn_coners(box_list, wlh_factor: float = 1.0):
    """
    frist we got List[Box], then we have to convert to List[coners]
    The result is Already Rotated!
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :return: corners_rotated: (N, 8, 3)
    """

    corners_list = np.array([box.corners(wlh_factor=wlh_factor).T for box in box_list])

    return corners_list


def filtrate_objects(obj_list, classes=None):
    """
    Discard objects which are not in self.classes (or its similar classes)
    :param obj_list: list
    :return: list
    """
    if classes is None:
        classes = ['car', 'bus', 'bicycle', 'emergency_vehicle', 'motorcycle', 'other_vehicle',
                   'pedestrian', 'truck']
    type_whitelist = classes
    # if mode == 'TRAIN' and cfg.INCLUDE_SIMILAR_TYPE:
    #     type_whitelist = list(self.classes)
    #     if 'Car' in self.classes:
    #         type_whitelist.append('Van')
    #     if 'Pedestrian' in self.classes:  # or 'Cyclist' in self.classes:
    #         type_whitelist.append('Person_sitting')

    valid_obj_list = []
    for obj in obj_list:
        if obj.name not in type_whitelist:  # rm Van, 20180928
            print("--------------There is animals?!------------------")
            continue
        # if self.mode == 'TRAIN' and cfg.PC_REDUCE_BY_RANGE and (self.check_pc_range(obj.pos) is False):
        #     continue
        valid_obj_list.append(obj)
    return valid_obj_list
