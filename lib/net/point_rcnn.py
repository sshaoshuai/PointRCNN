import torch
import torch.nn as nn
from lib.net.rpn import RPN
from lib.net.rcnn_net import RCNNNet
from lib.config import cfg


class PointRCNN(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TRAIN'):
        super().__init__()

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED:
            # need to add a watch
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)
        # 分阶段训练时，第一阶段不执行以下
        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features
            if cfg.RCNN.BACKBONE == 'pointnet':
                self.rcnn_net = RCNNNet(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'pointsift':
                raise NotImplementedError
            else:
                raise NotImplementedError

    def forward(self, input_data):
        # start with RPN.ENABLED, 可以包含两个阶段的处理
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            # 不固定权重参数且处于训练阶段才会回传梯度
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                # reference and no change weight
                if cfg.RPN.FIXED:
                    self.rpn.eval()  # Sets the module in evaluation mode.
                # 看ＲＰＮ代码，输出的结构和处理的方式
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)

            # rcnn inference, with prerequisite RPN.ENABLED
            if cfg.RCNN.ENABLED:
                with torch.no_grad():
                    # 取值
                    rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                    backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']

                    #
                    rpn_scores_raw = rpn_cls[:, :, 0]
                    # 归一化
                    rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                    # 筛选高于阈值的rpn
                    seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
                    pts_depth = torch.norm(backbone_xyz, p=2, dim=2)

                    # proposal layer
                    rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)

                    output['rois'] = rois
                    output['roi_scores_raw'] = roi_scores_raw
                    output['seg_result'] = seg_mask

                rcnn_input_info = {'rpn_xyz': backbone_xyz,
                                   'rpn_features': backbone_features.permute((0, 2, 1)),
                                   'seg_mask': seg_mask,
                                   'roi_boxes3d': rois,
                                   'pts_depth': pts_depth}
                if self.training:
                    rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']
                # 输出用于训练
                rcnn_output = self.rcnn_net(rcnn_input_info)
                output.update(rcnn_output)

        elif cfg.RCNN.ENABLED:  # 不从ＲＰＮ经过处理,直接对输出数据进行处理
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError

        return output
