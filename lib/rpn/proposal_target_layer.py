import torch
import torch.nn as nn
import numpy as np
from lib.config import cfg
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils


class ProposalTargetLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_dict):
        roi_boxes3d, gt_boxes3d = input_dict['roi_boxes3d'], input_dict['gt_boxes3d']

        batch_rois, batch_gt_of_rois, batch_roi_iou = self.sample_rois_for_rcnn(roi_boxes3d, gt_boxes3d)

        rpn_xyz, rpn_features = input_dict['rpn_xyz'], input_dict['rpn_features']
        if cfg.RCNN.USE_INTENSITY:
            pts_extra_input_list = [input_dict['rpn_intensity'].unsqueeze(dim=2),
                                    input_dict['seg_mask'].unsqueeze(dim=2)]
        else:
            pts_extra_input_list = [input_dict['seg_mask'].unsqueeze(dim=2)]

        if cfg.RCNN.USE_DEPTH:
            pts_depth = input_dict['pts_depth'] / 70.0 - 0.5
            pts_extra_input_list.append(pts_depth.unsqueeze(dim=2))
        pts_extra_input = torch.cat(pts_extra_input_list, dim=2)

        # point cloud pooling
        pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)
        pooled_features, pooled_empty_flag = \
            roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                          sampled_pt_num=cfg.RCNN.NUM_POINTS)

        sampled_pts, sampled_features = pooled_features[:, :, :, 0:3], pooled_features[:, :, :, 3:]

        # data augmentation
        if cfg.AUG_DATA:
            # data augmentation
            sampled_pts, batch_rois, batch_gt_of_rois = \
                self.data_augmentation(sampled_pts, batch_rois, batch_gt_of_rois)

        # canonical transformation
        batch_size = batch_rois.shape[0]
        roi_ry = batch_rois[:, :, 6] % (2 * np.pi)
        roi_center = batch_rois[:, :, 0:3]
        sampled_pts = sampled_pts - roi_center.unsqueeze(dim=2)  # (B, M, 512, 3)
        batch_gt_of_rois[:, :, 0:3] = batch_gt_of_rois[:, :, 0:3] - roi_center
        batch_gt_of_rois[:, :, 6] = batch_gt_of_rois[:, :, 6] - roi_ry

        for k in range(batch_size):
            sampled_pts[k] = kitti_utils.rotate_pc_along_y_torch(sampled_pts[k], batch_rois[k, :, 6])
            batch_gt_of_rois[k] = kitti_utils.rotate_pc_along_y_torch(batch_gt_of_rois[k].unsqueeze(dim=1),
                                                                      roi_ry[k]).squeeze(dim=1)

        # regression valid mask
        valid_mask = (pooled_empty_flag == 0)
        reg_valid_mask = ((batch_roi_iou > cfg.RCNN.REG_FG_THRESH) & valid_mask).long()

        # classification label
        batch_cls_label = (batch_roi_iou > cfg.RCNN.CLS_FG_THRESH).long()
        invalid_mask = (batch_roi_iou > cfg.RCNN.CLS_BG_THRESH) & (batch_roi_iou < cfg.RCNN.CLS_FG_THRESH)
        batch_cls_label[valid_mask == 0] = -1
        batch_cls_label[invalid_mask > 0] = -1

        output_dict = {'sampled_pts': sampled_pts.view(-1, cfg.RCNN.NUM_POINTS, 3),
                       'pts_feature': sampled_features.view(-1, cfg.RCNN.NUM_POINTS, sampled_features.shape[3]),
                       'cls_label': batch_cls_label.view(-1),
                       'reg_valid_mask': reg_valid_mask.view(-1),
                       'gt_of_rois': batch_gt_of_rois.view(-1, 7),
                       'gt_iou': batch_roi_iou.view(-1),
                       'roi_boxes3d': batch_rois.view(-1, 7)}

        return output_dict

    def sample_rois_for_rcnn(self, roi_boxes3d, gt_boxes3d):
        """
        :param roi_boxes3d: (B, M, 7)
        :param gt_boxes3d: (B, N, 8) [x, y, z, h, w, l, ry, cls]
        :return
            batch_rois: (B, N, 7)
            batch_gt_of_rois: (B, N, 8)
            batch_roi_iou: (B, N)
        """
        batch_size = roi_boxes3d.size(0)

        fg_rois_per_image = int(np.round(cfg.RCNN.FG_RATIO * cfg.RCNN.ROI_PER_IMAGE))

        batch_rois = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE, 7).zero_()
        batch_gt_of_rois = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE, 7).zero_()
        batch_roi_iou = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE).zero_()

        for idx in range(batch_size):
            cur_roi, cur_gt = roi_boxes3d[idx], gt_boxes3d[idx]

            k = cur_gt.__len__() - 1
            while cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]

            # include gt boxes in the candidate rois
            iou3d = iou3d_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)

            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

            # sample fg, easy_bg, hard_bg
            fg_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
            fg_inds = torch.nonzero((max_overlaps >= fg_thresh)).view(-1)

            # TODO: this will mix the fg and bg when CLS_BG_THRESH_LO < iou < CLS_BG_THRESH
            # fg_inds = torch.cat((fg_inds, roi_assignment), dim=0)  # consider the roi which has max_iou with gt as fg

            easy_bg_inds = torch.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH_LO)).view(-1)
            hard_bg_inds = torch.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH) &
                                         (max_overlaps >= cfg.RCNN.CLS_BG_THRESH_LO)).view(-1)

            fg_num_rois = fg_inds.numel()
            bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes3d).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE - fg_rois_per_this_image
                bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                rand_num = np.floor(np.random.rand(cfg.RCNN.ROI_PER_IMAGE) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes3d).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
                bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)

                fg_rois_per_this_image = 0
            else:
                import pdb
                pdb.set_trace()
                raise NotImplementedError

            # augment the rois by noise
            roi_list, roi_iou_list, roi_gt_list = [], [], []
            if fg_rois_per_this_image > 0:
                fg_rois_src = cur_roi[fg_inds]
                gt_of_fg_rois = cur_gt[gt_assignment[fg_inds]]
                iou3d_src = max_overlaps[fg_inds]
                fg_rois, fg_iou3d = self.aug_roi_by_noise_torch(fg_rois_src, gt_of_fg_rois, iou3d_src,
                                                                aug_times=cfg.RCNN.ROI_FG_AUG_TIMES)
                roi_list.append(fg_rois)
                roi_iou_list.append(fg_iou3d)
                roi_gt_list.append(gt_of_fg_rois)

            if bg_rois_per_this_image > 0:
                bg_rois_src = cur_roi[bg_inds]
                gt_of_bg_rois = cur_gt[gt_assignment[bg_inds]]
                iou3d_src = max_overlaps[bg_inds]
                aug_times = 1 if cfg.RCNN.ROI_FG_AUG_TIMES > 0 else 0
                bg_rois, bg_iou3d = self.aug_roi_by_noise_torch(bg_rois_src, gt_of_bg_rois, iou3d_src,
                                                                aug_times=aug_times)
                roi_list.append(bg_rois)
                roi_iou_list.append(bg_iou3d)
                roi_gt_list.append(gt_of_bg_rois)

            rois = torch.cat(roi_list, dim=0)
            iou_of_rois = torch.cat(roi_iou_list, dim=0)
            gt_of_rois = torch.cat(roi_gt_list, dim=0)

            batch_rois[idx] = rois
            batch_gt_of_rois[idx] = gt_of_rois
            batch_roi_iou[idx] = iou_of_rois

        return batch_rois, batch_gt_of_rois, batch_roi_iou

    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds, bg_rois_per_this_image):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = int(bg_rois_per_this_image * cfg.RCNN.HARD_BG_RATIO)
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    def aug_roi_by_noise_torch(self, roi_boxes3d, gt_boxes3d, iou3d_src, aug_times=10):
        iou_of_rois = torch.zeros(roi_boxes3d.shape[0]).type_as(gt_boxes3d)
        pos_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)

        for k in range(roi_boxes3d.shape[0]):
            temp_iou = cnt = 0
            roi_box3d = roi_boxes3d[k]

            gt_box3d = gt_boxes3d[k].view(1, 7)
            aug_box3d = roi_box3d
            keep = True
            while temp_iou < pos_thresh and cnt < aug_times:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d  # p=0.2 to keep the original roi box
                    keep = True
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
                    keep = False
                aug_box3d = aug_box3d.view((1, 7))
                iou3d = iou3d_utils.boxes_iou3d_gpu(aug_box3d, gt_box3d)
                temp_iou = iou3d[0][0]
                cnt += 1
            roi_boxes3d[k] = aug_box3d.view(-1)
            if cnt == 0 or keep:
                iou_of_rois[k] = iou3d_src[k]
            else:
                iou_of_rois[k] = temp_iou
        return roi_boxes3d, iou_of_rois

    @staticmethod
    def random_aug_box3d(box3d):
        """
        :param box3d: (7) [x, y, z, h, w, l, ry]
        random shift, scale, orientation
        """
        if cfg.RCNN.REG_AUG_METHOD == 'single':
            pos_shift = (torch.rand(3, device=box3d.device) - 0.5)  # [-0.5 ~ 0.5]
            hwl_scale = (torch.rand(3, device=box3d.device) - 0.5) / (0.5 / 0.15) + 1.0  #
            angle_rot = (torch.rand(1, device=box3d.device) - 0.5) / (0.5 / (np.pi / 12))  # [-pi/12 ~ pi/12]
            aug_box3d = torch.cat([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot], dim=0)
            return aug_box3d
        elif cfg.RCNN.REG_AUG_METHOD == 'multiple':
            # pos_range, hwl_range, angle_range, mean_iou
            range_config = [[0.2, 0.1, np.pi / 12, 0.7],
                            [0.3, 0.15, np.pi / 12, 0.6],
                            [0.5, 0.15, np.pi / 9, 0.5],
                            [0.8, 0.15, np.pi / 6, 0.3],
                            [1.0, 0.15, np.pi / 3, 0.2]]
            idx = torch.randint(low=0, high=len(range_config), size=(1,))[0].long()

            pos_shift = ((torch.rand(3, device=box3d.device) - 0.5) / 0.5) * range_config[idx][0]
            hwl_scale = ((torch.rand(3, device=box3d.device) - 0.5) / 0.5) * range_config[idx][1] + 1.0
            angle_rot = ((torch.rand(1, device=box3d.device) - 0.5) / 0.5) * range_config[idx][2]

            aug_box3d = torch.cat([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot], dim=0)
            return aug_box3d
        elif cfg.RCNN.REG_AUG_METHOD == 'normal':
            x_shift = np.random.normal(loc=0, scale=0.3)
            y_shift = np.random.normal(loc=0, scale=0.2)
            z_shift = np.random.normal(loc=0, scale=0.3)
            h_shift = np.random.normal(loc=0, scale=0.25)
            w_shift = np.random.normal(loc=0, scale=0.15)
            l_shift = np.random.normal(loc=0, scale=0.5)
            ry_shift = ((torch.rand() - 0.5) / 0.5) * np.pi / 12

            aug_box3d = np.array([box3d[0] + x_shift, box3d[1] + y_shift, box3d[2] + z_shift, box3d[3] + h_shift,
                                  box3d[4] + w_shift, box3d[5] + l_shift, box3d[6] + ry_shift], dtype=np.float32)
            aug_box3d = torch.from_numpy(aug_box3d).type_as(box3d)
            return aug_box3d
        else:
            raise NotImplementedError

    def data_augmentation(self, pts, rois, gt_of_rois):
        """
        :param pts: (B, M, 512, 3)
        :param rois: (B, M. 7)
        :param gt_of_rois: (B, M, 7)
        :return:
        """
        batch_size, boxes_num = pts.shape[0], pts.shape[1]

        # rotation augmentation
        angles = (torch.rand((batch_size, boxes_num), device=pts.device) - 0.5 / 0.5) * (np.pi / cfg.AUG_ROT_RANGE)

        # calculate gt alpha from gt_of_rois
        temp_x, temp_z, temp_ry = gt_of_rois[:, :, 0], gt_of_rois[:, :, 2], gt_of_rois[:, :, 6]
        temp_beta = torch.atan2(temp_z, temp_x)
        gt_alpha = -torch.sign(temp_beta) * np.pi / 2 + temp_beta + temp_ry  # (B, M)

        temp_x, temp_z, temp_ry = rois[:, :, 0], rois[:, :, 2], rois[:, :, 6]
        temp_beta = torch.atan2(temp_z, temp_x)
        roi_alpha = -torch.sign(temp_beta) * np.pi / 2 + temp_beta + temp_ry  # (B, M)

        for k in range(batch_size):
            pts[k] = kitti_utils.rotate_pc_along_y_torch(pts[k], angles[k])
            gt_of_rois[k] = kitti_utils.rotate_pc_along_y_torch(gt_of_rois[k].unsqueeze(dim=1), angles[k]).squeeze(dim=1)
            rois[k] = kitti_utils.rotate_pc_along_y_torch(rois[k].unsqueeze(dim=1), angles[k]).squeeze(dim=1)

            # calculate the ry after rotation
            temp_x, temp_z = gt_of_rois[:, :, 0], gt_of_rois[:, :, 2]
            temp_beta = torch.atan2(temp_z, temp_x)
            gt_of_rois[:, :, 6] = torch.sign(temp_beta) * np.pi / 2 + gt_alpha - temp_beta

            temp_x, temp_z = rois[:, :, 0], rois[:, :, 2]
            temp_beta = torch.atan2(temp_z, temp_x)
            rois[:, :, 6] = torch.sign(temp_beta) * np.pi / 2 + roi_alpha - temp_beta

        # scaling augmentation
        scales = 1 + ((torch.rand((batch_size, boxes_num), device=pts.device) - 0.5) / 0.5) * 0.05
        pts = pts * scales.unsqueeze(dim=2).unsqueeze(dim=3)
        gt_of_rois[:, :, 0:6] = gt_of_rois[:, :, 0:6] * scales.unsqueeze(dim=2)
        rois[:, :, 0:6] = rois[:, :, 0:6] * scales.unsqueeze(dim=2)

        # flip augmentation
        flip_flag = torch.sign(torch.rand((batch_size, boxes_num), device=pts.device) - 0.5)
        pts[:, :, :, 0] = pts[:, :, :, 0] * flip_flag.unsqueeze(dim=2)
        gt_of_rois[:, :, 0] = gt_of_rois[:, :, 0] * flip_flag
        # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
        src_ry = gt_of_rois[:, :, 6]
        ry = (flip_flag == 1).float() * src_ry + (flip_flag == -1).float() * (torch.sign(src_ry) * np.pi - src_ry)
        gt_of_rois[:, :, 6] = ry

        rois[:, :, 0] = rois[:, :, 0] * flip_flag
        # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
        src_ry = rois[:, :, 6]
        ry = (flip_flag == 1).float() * src_ry + (flip_flag == -1).float() * (torch.sign(src_ry) * np.pi - src_ry)
        rois[:, :, 6] = ry

        return pts, rois, gt_of_rois
