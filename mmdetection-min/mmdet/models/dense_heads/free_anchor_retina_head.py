# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdet.core import bbox_overlaps
from ..builder import HEADS
from .retina_head import RetinaHead

EPS = 1e-12


@HEADS.register_module()
class FreeAnchorRetinaHead(RetinaHead):
    """FreeAnchor RetinaHead used in https://arxiv.org/abs/1909.02466.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32,
            requires_grad=True).
        pre_anchor_topk (int): Number of boxes that be token in each bag.
        bbox_thr (float): The threshold of the saturated linear function. It is
            usually the same with the IoU threshold used in NMS.
        gamma (float): Gamma parameter in focal loss.
        alpha (float): Alpha parameter in focal loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 pre_anchor_topk=50,
                 bbox_thr=0.6,
                 gamma=2.0,
                 alpha=0.5,
                 **kwargs):
        super(FreeAnchorRetinaHead,
              self).__init__(num_classes, in_channels, stacked_convs, conv_cfg,
                             norm_cfg, **kwargs)

        self.pre_anchor_topk = pre_anchor_topk
        self.bbox_thr = bbox_thr
        self.gamma = gamma
        self.alpha = alpha

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]  # FPN每个输出的尺度大小
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list, _ = self.get_anchors(featmap_sizes, img_metas, device=device)  # 生成先验框(每个网格存在9个先验框)
        anchors = [torch.cat(anchor) for anchor in anchor_list]  # 把每张图片的所有的先验框拼接起来

        # concatenate each level
        cls_scores = [
            cls.permute(0, 2, 3, 1).reshape(cls.size(0), -1, self.cls_out_channels)
            for cls in cls_scores
        ]
        bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(bbox_pred.size(0), -1, 4)
            for bbox_pred in bbox_preds
        ]
        # 每张图片所有的分类和回归预测
        cls_scores = torch.cat(cls_scores, dim=1)  # [BS, N, num_classes]
        bbox_preds = torch.cat(bbox_preds, dim=1)  # [BS, N, 4]

        cls_prob = torch.sigmoid(cls_scores)  # [BS, N, num_classes]
        box_prob = []
        num_pos = 0  # 真实样本数量
        positive_losses = []
        # [N, 4], [num_gts,], [num_gts, 4], [N, num_classes], [N, 4]
        for _, (anchors_, gt_labels_, gt_bboxes_, cls_prob_, bbox_preds_) in enumerate(
                zip(anchors, gt_labels, gt_bboxes, cls_prob, bbox_preds)):

            with torch.no_grad():
                if len(gt_bboxes_) == 0:
                    image_box_prob = torch.zeros(anchors_.size(0), self.cls_out_channels).type_as(bbox_preds_)
                else:
                    # box_localization: a_{j}^{loc}, shape: [j, 4]
                    # 先验框解码得到预测框
                    pred_boxes = self.bbox_coder.decode(anchors_, bbox_preds_)

                    # object_box_iou: IoU_{ij}^{loc}, shape: [i, j]
                    # 所有真实框与所有预测框计算IoU [num_gts, N]
                    object_box_iou = bbox_overlaps(gt_bboxes_, pred_boxes)

                    # object_box_prob: P{a_{j} -> b_{i}}, shape: [i, j]
                    t1 = self.bbox_thr
                    # 每个真实框与所有预测框的最大IoU
                    t2 = object_box_iou.max(dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)  # [num_gts, 1]

                    # 预测框匹配真实框的概率
                    # Saturated linear
                    # https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652053346&idx=5&sn=7502bfefe636115999837c4fca9f93be
                    object_box_prob = ((object_box_iou - t1) / (t2 - t1)).clamp(min=0, max=1)  # [num_gts, N]

                    # object_cls_box_prob: P{a_{j} -> b_{i}}, shape: [i, c, j]
                    # 图中物体数量
                    num_obj = gt_labels_.size(0)
                    # 为物体类别添加一个索引 [2, num_gts]: [[0, 1, 2, 3, ...], [9, 5, 4, 9, ...]]
                    indices = torch.stack([torch.arange(num_obj).type_as(gt_labels_), gt_labels_], dim=0)
                    #  [num_gts, max(indices[1:]) + 1, N]: [物体数量, 最大类别 + 1, 预测框数量]
                    # indices为索引 object_box_prob为与索引相对应的值
                    object_cls_box_prob = torch.sparse_coo_tensor(indices, object_box_prob)

                    # image_box_iou: P{a_{j} \in A_{+}}, shape: [c, j]
                    """
                    from "start" to "end" implement:
                    image_box_iou = torch.sparse.max(object_cls_box_prob,im=0).t()
                    """
                    # start
                    # 所有预测框对应 物体真实框 的目标框概率值, 不存在默认为0
                    box_cls_prob = torch.sparse.sum(object_cls_box_prob, dim=0).to_dense()  # [max(indices[1:]) + 1, N]
                    # 所有预测框对应 物体真实框 的目标框概率值非0的索引 (注: 部分的真实框可能没有对应满足条件的预测框)
                    indices = torch.nonzero(box_cls_prob, as_tuple=False).t_()  # 类别, 预测框索引
                    if indices.numel() == 0:
                        image_box_prob = torch.zeros(anchors_.size(0), self.cls_out_channels).type_as(object_box_prob)
                    else:
                        # 将indices筛选结果按照真实物体进行规整
                        nonzero_box_prob = torch.where(
                            (gt_labels_.unsqueeze(dim=-1) == indices[0]),
                            object_box_prob[:, indices[1]],  # n个预测框相对每个真实框的object_box_prob [num_gts, n]
                            torch.tensor([0]).type_as(object_box_prob)
                        ).max(dim=0).values

                        # upmap to shape [j, c]
                        # 将上述结果映射成先验框的形式
                        image_box_prob = torch.sparse_coo_tensor(
                            indices.flip([0]),
                            nonzero_box_prob,
                            size=(anchors_.size(0), self.cls_out_channels)).to_dense()
                    # end

                box_prob.append(image_box_prob)

            # construct bags for objects
            # 真实框和先验框计算IoU
            match_quality_matrix = bbox_overlaps(gt_bboxes_, anchors_)  # [num_gts, N]
            # 取出与每个真实框前 self.pre_anchor_topk 个 IoU 较大的先验框索引  [num_gts, self.pre_anchor_topk]
            _, matched = torch.topk(
                match_quality_matrix,
                self.pre_anchor_topk,
                dim=1,
                sorted=False)
            del match_quality_matrix

            # matched_cls_prob: P_{ij}^{cls}
            # cls_prob_[matched]: [num_gts, pre_anchor_topk, num_classes] 先验框索引匹配的分类预测 (S激活后的)
            # gt_labels_: [num_gts, ] -> [num_gts, 1, 1] -> [num_gts, self.pre_anchor_topk, 1]
            # matched_cls_prob: [num_gts, pre_anchor_topk]
            # 取出满足要求先验框的分类预测
            matched_cls_prob = torch.gather(
                cls_prob_[matched], 2,
                gt_labels_.view(-1, 1, 1).repeat(1, self.pre_anchor_topk, 1)).squeeze(2)

            # matched_box_prob: P_{ij}^{loc}
            matched_anchors = anchors_[matched]  # [num_gts, pre_anchor_topk, 4] 取出满足要求的先验框
            matched_object_targets = self.bbox_coder.encode(  # 先验框与其负责预测真实框进行编码
                matched_anchors,
                gt_bboxes_.unsqueeze(dim=1).expand_as(matched_anchors))  # [num_gts, pre_anchor_topk, (dx, dy, dw, dh)]

            loss_bbox = self.loss_bbox(  # [num_gts, pre_anchor_topk]
                bbox_preds_[matched],
                matched_object_targets,
                reduction_override='none').sum(-1)
            matched_box_prob = torch.exp(-loss_bbox)  # [num_gts, pre_anchor_topk]

            # positive_losses: {-log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) )}
            num_pos += len(gt_bboxes_)
            positive_losses.append(self.positive_bag_loss(matched_cls_prob, matched_box_prob))

        # ------------------------------------------------------------------------------------------------------- #
        positive_loss = torch.cat(positive_losses).sum() / max(1, num_pos)

        # box_prob: P{a_{j} \in A_{+}}
        box_prob = torch.stack(box_prob, dim=0)

        # negative_loss:
        # \sum_{j}{ FL((1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg})) } / n||B||
        negative_loss = self.negative_bag_loss(cls_prob, box_prob).sum() / max(1, num_pos * self.pre_anchor_topk)

        # avoid the absence of gradients in regression subnet
        # when no ground-truth in a batch
        if num_pos == 0:
            positive_loss = bbox_preds.sum() * 0

        losses = {
            'positive_bag_loss': positive_loss,
            'negative_bag_loss': negative_loss
        }
        return losses

    def positive_bag_loss(self, matched_cls_prob, matched_box_prob):
        """Compute positive bag loss.

        :math:`-log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) )`.

        :math:`P_{ij}^{cls}`: matched_cls_prob, classification probability of matched samples.

        :math:`P_{ij}^{loc}`: matched_box_prob, box probability of matched samples.

        Args:
            matched_cls_prob (Tensor): Classification probability of matched
                samples in shape (num_gt, pre_anchor_topk).
            matched_box_prob (Tensor): BBox probability of matched samples,
                in shape (num_gt, pre_anchor_topk).

        Returns:
            Tensor: Positive bag loss in shape (num_gt,).
        """  # noqa: E501, W605
        # bag_prob = Mean-max(matched_prob)
        matched_prob = matched_cls_prob * matched_box_prob
        weight = 1 / torch.clamp(1 - matched_prob, 1e-12, None)
        weight /= weight.sum(dim=1).unsqueeze(dim=-1)
        bag_prob = (weight * matched_prob).sum(dim=1)
        # positive_bag_loss = -self.alpha * log(bag_prob)
        return self.alpha * F.binary_cross_entropy(bag_prob, torch.ones_like(bag_prob), reduction='none')

    def negative_bag_loss(self, cls_prob, box_prob):
        """Compute negative bag loss.

        :math:`FL((1 - P_{a_{j} \in A_{+}}) * (1 - P_{j}^{bg}))`.

        :math:`P_{a_{j} \in A_{+}}`: Box_probability of matched samples.

        :math:`P_{j}^{bg}`: Classification probability of negative samples.

        Args:
            cls_prob (Tensor): Classification probability, in shape
                (num_img, num_anchors, num_classes).
            box_prob (Tensor): Box probability, in shape
                (num_img, num_anchors, num_classes).

        Returns:
            Tensor: Negative bag loss in shape (num_img, num_anchors, num_classes).
        """  # noqa: E501, W605
        prob = cls_prob * (1 - box_prob)
        # There are some cases when neg_prob = 0.
        # This will cause the neg_prob.log() to be inf without clamp.
        prob = prob.clamp(min=EPS, max=1 - EPS)
        negative_bag_loss = prob ** self.gamma * F.binary_cross_entropy(prob, torch.zeros_like(prob), reduction='none')
        return (1 - self.alpha) * negative_bag_loss
