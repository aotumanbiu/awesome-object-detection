# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class ATSSAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    If ``alpha`` is not None, it means that the dynamic cost
    ATSSAssigner is adopted, which is currently only used in the DDOD.

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 topk,
                 alpha=None,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.alpha = alpha
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    """Assign a corresponding gt bbox or background to each bbox.

    Args:
        topk (int): number of bbox selected in each level.
        alpha (float): param of cost rate for each proposal only in DDOD.
            Default None.
        iou_calculator (dict): builder of IoU calculator.
            Default dict(type='BboxOverlaps2D').
        ignore_iof_thr (int): whether ignore max overlaps or not.
            Default -1 (1 or -1).
    """

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py
    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               cls_scores=None,
               bbox_preds=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt

        If ``alpha`` is not None, and ``cls_scores`` and `bbox_preds`
        are not None, the overlaps calculation in the first step
        will also include dynamic cost, which is currently only used in
        the DDOD.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO. Default None.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).
            cls_scores (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes. Default None.
            bbox_preds (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4. Default None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """

        ## 注意: FCOS中 每个网格只有一个先验框

        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        message = 'Invalid alpha parameter because cls_scores or ' \
                  'bbox_preds are None. If you want to use the ' \
                  'cost-based ATSSAssigner,  please set cls_scores, ' \
                  'bbox_preds and self.alpha at the same time. '

        if self.alpha is None:
            # ATSSAssigner
            overlaps = self.iou_calculator(bboxes, gt_bboxes)  # 计算先验框和真实框IoU [num_bboxes, num_gt]
            if cls_scores is not None or bbox_preds is not None:
                warnings.warn(message)
        else:
            # Dynamic cost ATSSAssigner in DDOD
            assert cls_scores is not None and bbox_preds is not None, message

            # compute cls cost for bbox and GT
            cls_cost = torch.sigmoid(cls_scores[:, gt_labels])

            # compute iou between all bbox and gt
            overlaps = self.iou_calculator(bbox_preds, gt_bboxes)

            # make sure that we are in element-wise multiplication
            assert cls_cost.shape == overlaps.shape

            # overlaps is actually a cost matrix
            overlaps = cls_cost ** (1 - self.alpha) * overlaps ** self.alpha

        # assign 0 by default
        # 存放先验框所匹配真实框的索引
        assigned_gt_inds = overlaps.new_full((num_bboxes,), 0, dtype=torch.long)  # [num_bboxes, ]

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)

            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)  # 真实框中心点坐标 [num_gt, 2]

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)  # 先验框中心点坐标 [num_bboxes, 2]

        # 先验框到真实框的中心点距离r
        distances = (bboxes_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()  # [num_bboxes, num_gt]

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []  # 存放先验框候选点索引
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):  # 每层先验筛选出k个与真实框距离最小的先验框索引
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)

            _, topk_idxs_per_level = distances_per_level.topk(selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx

        # 注意: 这里先验框可以匹配多个真实框, 因此会出现重复的索引
        # 因此满足要求的先验框数量最大不超过 k * num_level_bboxes * num_gts
        candidate_idxs = torch.cat(candidate_idxs, dim=0)  # [k * num_level_bboxes , num_gt]

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        # 取出候选点与真实框的IoU [k * num_level_bboxes , num_gt]
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        # 与某真实框匹配的所有先验框的IoU均值和方差
        overlaps_mean_per_gt = candidate_overlaps.mean(0)  # [num_gt, ]
        overlaps_std_per_gt = candidate_overlaps.std(0)  # [num_gt, ]
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        # 与某真实框匹配的所有先验框的IoU大于上述overlaps_thr_per_gt则为正样本
        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        # 将每列存取的索引值往后平移num_bboxes位置
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes

        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)  # [num_bboxes * num_gt, ]
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)  # [num_bboxes * num_gt]
        candidate_idxs = candidate_idxs.view(-1)  # [num_gt * k * num_level_bboxes , ]

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        # 因为candidate_idxs存放索引值的平移 以及 ep_bboxes_cx维度扩展和展平
        # 使其刚好能后取得所对应的中心点坐标
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01  # 确保先验框中心点在真实框内部

        # 第一步筛选 IoU大于阈值
        # 第二步筛选 先验框中心点在真实框内部
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        # [num_bboxes, num_gt] -> [num_gt, num_bboxes] -> [num_bboxes * num_gt, ]
        # 从上面的维度变换可以看出其与candidate_idxs对齐了
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)

        # 满足要求的索引值所对应位置IoU不变, 否则为负无穷大-INF
        index = candidate_idxs.view(-1)[is_pos.view(-1)]  # 取出最终满足上述两个条件所存放的索引值
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        # [num_gt, ], [num_gt, 0] -> 每个真实框与所有先验的IoU的最大值, 最大IoU所对应先验框的索引
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        # [num_bboxes, ] 每个先验框所负责预测真实框的 索引值 + 1
        # 例如第一先验负责预测第一个真实框, 那么就会存放索引值0, 但这里存放的是 0 + 1 = 1
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            # 存放每个网格所负责预测的类别
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()  # 取出正样本先验框的索引
            if pos_inds.numel() > 0:  # 正样本数量是否大于0
                # 正样本先验框的类别 负样本则为-1
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
