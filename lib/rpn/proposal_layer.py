import torch
import torch.nn as nn
from lib.utils.bbox_transform import decode_bbox_target
from lib.config import cfg
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils


class ProposalLayer(nn.Module):
    def __init__(self, mode='TRAIN'):
        super().__init__()
        self.mode = mode
        self.MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

    def forward(self, rpn_scores, rpn_reg, xyz):
        """
        :param rpn_scores: (B, N)
        :param rpn_reg: (B, N, 8)
        :param xyz: (B, N, 3)
        :return bbox3d: (B, M, 7)
        """
        batch_size = xyz.shape[0]
        proposals = decode_bbox_target(xyz.view(-1, 3), rpn_reg.view(-1, rpn_reg.shape[-1]),
                                       anchor_size=self.MEAN_SIZE,
                                       loc_scope=cfg.RPN.LOC_SCOPE,
                                       loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                                       num_head_bin=cfg.RPN.NUM_HEAD_BIN,
                                       get_xz_fine=cfg.RPN.LOC_XZ_FINE,
                                       get_y_by_bin=False,
                                       get_ry_fine=False)  # (N, 7)
        proposals[:, 1] += proposals[:, 3] / 2  # set y as the center of bottom
        proposals = proposals.view(batch_size, -1, 7)

        scores = rpn_scores
        _, sorted_idxs = torch.sort(scores, dim=1, descending=True)

        batch_size = scores.size(0)
        ret_bbox3d = scores.new(batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N, 7).zero_()
        ret_scores = scores.new(batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N).zero_()
        for k in range(batch_size):
            scores_single = scores[k]
            proposals_single = proposals[k]
            order_single = sorted_idxs[k]

            if cfg.TEST.RPN_DISTANCE_BASED_PROPOSE:
                scores_single, proposals_single = self.distance_based_proposal(scores_single, proposals_single,
                                                                               order_single)
            else:
                scores_single, proposals_single = self.score_based_proposal(scores_single, proposals_single,
                                                                            order_single)

            proposals_tot = proposals_single.size(0)
            ret_bbox3d[k, :proposals_tot] = proposals_single
            ret_scores[k, :proposals_tot] = scores_single

        return ret_bbox3d, ret_scores

    def distance_based_proposal(self, scores, proposals, order):
        """
         propose rois in two area based on the distance
        :param scores: (N)
        :param proposals: (N, 7)
        :param order: (N)
        """
        nms_range_list = [0, 40.0, 80.0]
        pre_tot_top_n = cfg[self.mode].RPN_PRE_NMS_TOP_N
        pre_top_n_list = [0, int(pre_tot_top_n * 0.7), pre_tot_top_n - int(pre_tot_top_n * 0.7)]
        post_tot_top_n = cfg[self.mode].RPN_POST_NMS_TOP_N
        post_top_n_list = [0, int(post_tot_top_n * 0.7), post_tot_top_n - int(post_tot_top_n * 0.7)]

        scores_single_list, proposals_single_list = [], []

        # sort by score
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]

        dist = proposals_ordered[:, 2]
        first_mask = (dist > nms_range_list[0]) & (dist <= nms_range_list[1])
        for i in range(1, len(nms_range_list)):
            # get proposal distance mask
            dist_mask = ((dist > nms_range_list[i - 1]) & (dist <= nms_range_list[i]))

            if dist_mask.sum() != 0:
                # this area has points
                # reduce by mask
                cur_scores = scores_ordered[dist_mask]
                cur_proposals = proposals_ordered[dist_mask]

                # fetch pre nms top K
                cur_scores = cur_scores[:pre_top_n_list[i]]
                cur_proposals = cur_proposals[:pre_top_n_list[i]]
            else:
                assert i == 2, '%d' % i
                # this area doesn't have any points, so use rois of first area
                cur_scores = scores_ordered[first_mask]
                cur_proposals = proposals_ordered[first_mask]

                # fetch top K of first area
                cur_scores = cur_scores[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]
                cur_proposals = cur_proposals[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]

            # oriented nms
            boxes_bev = kitti_utils.boxes3d_to_bev_torch(cur_proposals)
            if cfg.RPN.NMS_TYPE == 'rotate':
                keep_idx = iou3d_utils.nms_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
            elif cfg.RPN.NMS_TYPE == 'normal':
                keep_idx = iou3d_utils.nms_normal_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
            else:
                raise NotImplementedError

            # Fetch post nms top k
            keep_idx = keep_idx[:post_top_n_list[i]]

            scores_single_list.append(cur_scores[keep_idx])
            proposals_single_list.append(cur_proposals[keep_idx])

        scores_single = torch.cat(scores_single_list, dim=0)
        proposals_single = torch.cat(proposals_single_list, dim=0)
        return scores_single, proposals_single

    def score_based_proposal(self, scores, proposals, order):
        """
         propose rois in two area based on the distance
        :param scores: (N)
        :param proposals: (N, 7)
        :param order: (N)
        """
        # sort by score
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]

        # pre nms top K
        cur_scores = scores_ordered[:cfg[self.mode].RPN_PRE_NMS_TOP_N]
        cur_proposals = proposals_ordered[:cfg[self.mode].RPN_PRE_NMS_TOP_N]

        boxes_bev = kitti_utils.boxes3d_to_bev_torch(cur_proposals)
        keep_idx = iou3d_utils.nms_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)

        # Fetch post nms top k
        keep_idx = keep_idx[:cfg[self.mode].RPN_POST_NMS_TOP_N]

        return cur_scores[keep_idx], cur_proposals[keep_idx]



