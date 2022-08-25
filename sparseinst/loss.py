# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment
from fvcore.nn import sigmoid_focal_loss_jit

from detectron2.utils.registry import Registry

from .utils import nested_masks_from_list, is_dist_avail_and_initialized, get_world_size, box_cxcywh_to_xyxy, generalized_box_iou

SPARSE_INST_MATCHER_REGISTRY = Registry("SPARSE_INST_MATCHER")
SPARSE_INST_MATCHER_REGISTRY.__doc__ = "Matcher for SparseInst"
SPARSE_INST_CRITERION_REGISTRY = Registry("SPARSE_INST_CRITERION")
SPARSE_INST_CRITERION_REGISTRY.__doc__ = "Criterion for SparseInst"


def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.4).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_score(inputs, targets):
    inputs = inputs.sigmoid()
    numerator = 2 * torch.matmul(inputs, targets.t())
    denominator = (
        inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


def dice_loss(inputs, targets, reduction='sum'):
    inputs = inputs.sigmoid()
    assert inputs.shape == targets.shape
    numerator = 2 * (inputs * targets).sum(1)
    denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
    loss = 1 - (numerator) / (denominator + 1e-4)
    if reduction == 'none':
        return loss
    return loss.sum()


@SPARSE_INST_CRITERION_REGISTRY.register()
class SparseInstCriterion(nn.Module):
    # This part is partially derivated from: https://github.com/facebookresearch/detr/blob/main/models/detr.py

    def __init__(self, cfg, matcher):
        super().__init__()
        self.matcher = matcher
        self.losses = cfg.MODEL.SPARSE_INST.LOSS.ITEMS
        self.weight_dict = self.get_weight_dict(cfg)
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = 0.1
        self.register_buffer('empty_weight', empty_weight)

    def get_weight_dict(self, cfg):
        losses = ("loss_ce", "loss_bbox", "loss_giou")
        weight_dict = {}
        ce_weight = cfg.MODEL.SPARSE_INST.LOSS.CLASS_WEIGHT
        bbox_weight = cfg.MODEL.SPARSE_INST.LOSS.MASK_BBOX_WEIGHT
        giou_weight = cfg.MODEL.SPARSE_INST.LOSS.MASK_GIOU_WEIGHT


        weight_dict = dict(
            zip(losses, (ce_weight, bbox_weight, giou_weight)))
        return weight_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, num_instances, input_shape=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J]
                                     for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_instances, input_shape=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_instances

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_instances   
        return losses
    
    # def loss_masks_with_iou_objectness(self, outputs, targets, indices, num_instances, input_shape):
    #     src_idx = self._get_src_permutation_idx(indices)
    #     tgt_idx = self._get_tgt_permutation_idx(indices)
    #     # Bx100xHxW
    #     assert "pred_masks" in outputs
    #     assert "pred_scores" in outputs
    #     src_iou_scores = outputs["pred_scores"]
    #     src_masks = outputs["pred_masks"]
    #     with torch.no_grad():
    #         target_masks, _ = nested_masks_from_list(
    #             [t["masks"].tensor for t in targets], input_shape).decompose()
    #     num_masks = [len(t["masks"]) for t in targets]
    #     target_masks = target_masks.to(src_masks)
    #     if len(target_masks) == 0:
    #         losses = {
    #             "loss_dice": src_masks.sum() * 0.0,
    #             "loss_mask": src_masks.sum() * 0.0,
    #             "loss_objectness": src_iou_scores.sum() * 0.0
    #         }
    #         return losses

    #     src_masks = src_masks[src_idx]
    #     target_masks = F.interpolate(
    #         target_masks[:, None], size=src_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

    #     src_masks = src_masks.flatten(1)
    #     # FIXME: tgt_idx
    #     mix_tgt_idx = torch.zeros_like(tgt_idx[1])
    #     cum_sum = 0
    #     for num_mask in num_masks:
    #         mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
    #         cum_sum += num_mask
    #     mix_tgt_idx += tgt_idx[1]

    #     target_masks = target_masks[mix_tgt_idx].flatten(1)

    #     with torch.no_grad():
    #         ious = compute_mask_iou(src_masks, target_masks)

    #     tgt_iou_scores = ious
    #     src_iou_scores = src_iou_scores[src_idx]
    #     tgt_iou_scores = tgt_iou_scores.flatten(0)
    #     src_iou_scores = src_iou_scores.flatten(0)

    #     losses = {
    #         "loss_objectness": F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean'),
    #         "loss_dice": dice_loss(src_masks, target_masks) / num_instances,
    #         "loss_mask": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean')
    #     }
    #     return losses

    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        if loss == "loss_objectness":
            # NOTE: loss_objectness will be calculated in `loss_masks_with_iou_objectness`
            return {}
        assert loss in loss_map
        return loss_map[loss](outputs, targets, indices, num_instances, **kwargs)

    def forward(self, outputs, targets, input_shape):

        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, input_shape)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = torch.as_tensor(
            [num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_instances)
        num_instances = torch.clamp(
            num_instances / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices,
                                        num_instances, input_shape=input_shape))

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return losses


@SPARSE_INST_MATCHER_REGISTRY.register()
class SparseInstMatcherV1(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.alpha = cfg.MODEL.SPARSE_INST.MATCHER.ALPHA
        self.beta = cfg.MODEL.SPARSE_INST.MATCHER.BETA
        self.mask_score = dice_score

    @torch.no_grad()
    def forward(self, outputs, targets, input_shape):
        B, N, H, W = outputs["pred_masks"].shape
        pred_masks = outputs['pred_masks']
        pred_logits = outputs['pred_logits'].sigmoid()

        indices = []

        for i in range(B):
            tgt_ids = targets[i]["labels"]
            # no annotations
            if tgt_ids.shape[0] == 0:
                indices.append((torch.as_tensor([]),
                                torch.as_tensor([])))
                continue

            tgt_masks = targets[i]['masks'].tensor.to(pred_masks)
            pred_logit = pred_logits[i]
            out_masks = pred_masks[i]

            # upsampling:
            # (1) padding/
            # (2) upsampling to 1x input size (input_shape)
            # (3) downsampling to 0.25x input size (output mask size)
            ori_h, ori_w = tgt_masks.size(1), tgt_masks.size(2)
            tgt_masks_ = torch.zeros(
                (1, tgt_masks.size(0), input_shape[0], input_shape[1])).to(pred_masks)
            tgt_masks_[0, :, :ori_h, :ori_w] = tgt_masks
            tgt_masks = F.interpolate(
                tgt_masks_, size=out_masks.shape[-2:], mode='bilinear', align_corners=False)[0]

            # compute dice score and classification score
            tgt_masks = tgt_masks.flatten(1)
            out_masks = out_masks.flatten(1)

            mask_score = self.mask_score(out_masks, tgt_masks)
            # Nx(Number of gts)
            matching_prob = pred_logit[:, tgt_ids]
            C = (mask_score ** self.alpha) * (matching_prob ** self.beta)
            # hungarian matching
            inds = linear_sum_assignment(C.cpu(), maximize=True)
            indices.append(inds)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


@SPARSE_INST_MATCHER_REGISTRY.register()
class SparseInstMatcher(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cost_class =  cfg.MODEL.SPARSE_INST.MATCHER.COST_CLASS
        self.cost_bbox = cfg.MODEL.SPARSE_INST.MATCHER.COST_BBOX
        self.cost_giou = cfg.MODEL.SPARSE_INST.MATCHER.COST_GIOU

    def forward(self, outputs, targets, input_shape):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            if tgt_ids.shape[0] == 0:
                return [(torch.as_tensor([]).to(out_prob), torch.as_tensor([]).to(out_prob))] * bs

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_sparse_inst_matcher(cfg):
    name = cfg.MODEL.SPARSE_INST.MATCHER.NAME
    return SPARSE_INST_MATCHER_REGISTRY.get(name)(cfg)


def build_sparse_inst_criterion(cfg):
    matcher = build_sparse_inst_matcher(cfg)
    name = cfg.MODEL.SPARSE_INST.LOSS.NAME
    return SPARSE_INST_CRITERION_REGISTRY.get(name)(cfg, matcher)
