from operator import itemgetter
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

import torch

from torch import nn, Tensor
import torch.nn.functional as F

from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import (
    RegionProposalNetwork,
    concat_box_prediction_layers,
)
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import boxes as box_ops


class XRegionProposalNetwork(RegionProposalNetwork):
    def __init__(self, source):
        self.__dict__.update(source.__dict__)

    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: List[Tuple[int, int]],
        num_anchors_per_level: List[int],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(
            proposals, objectness_prob, levels, image_shapes
        ):
            lvl = torch.zeros_like(lvl)
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def forward(
        self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        # modified to also return objectness score
        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [
            s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors
        ]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(
            objectness, pred_bbox_deltas
        )
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(
            proposals, objectness, images.image_sizes, num_anchors_per_level
        )

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return {"boxes": boxes, "scores": scores, "losses": losses}


class XRoIHeads(RoIHeads):
    def __init__(self, source):
        self.__dict__.update(source.__dict__)

    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        # so far the same as the postprocess_method from the torchvision impl.

        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):

        # modified to return box features
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert (
                    t["boxes"].dtype in floating_point_types
                ), "target boxes must of float type"
                assert (
                    t["labels"].dtype == torch.int64
                ), "target labels must of int64 type"
                if self.has_keypoint():
                    assert (
                        t["keypoints"].dtype == torch.float32
                    ), "target keypoints must of float type"

        if self.training:
            (
                proposals,
                matched_idxs,
                labels,
                regression_targets,
            ) = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features0 = self.box_roi_pool(features, proposals, image_shapes)

        ## box_featuers0 is a 256x7x7 (the 7x7 is a parameter from the box_roi_palign).
        # it is then flattened to a single 1024 vector by the box_head
        box_features = self.box_head(box_features0)

        # there is a final box for each class (it is category dependent).
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )

            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return {"result": result, "losses": losses, "box_features": box_features}


class XGeneralizedRCNN(GeneralizedRCNN):
    def __init__(self, source):
        self.__dict__.update(source.__dict__)
        self.rpn = XRegionProposalNetwork(self.rpn)
        self.roi_heads = XRoIHeads(self.roi_heads)

    def forward(self, images, targets=None):
        # modify to use modified RPN, and roi_heads
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}."
                        )
                else:
                    raise ValueError(
                        f"Expected target boxes to be of type Tensor, got {type(boxes)}."
                    )

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # proposals, proposal_losses = self.rpn(images, features, targets)
        rpn_out = self.rpn(images, features, targets)
        proposals, proposal_losses, objectness_scores = itemgetter(
            "boxes", "losses", "scores"
        )(rpn_out)

        # transpose order: key, index -> order: index, key, just like detections
        rpn_out_transpose = []
        for i in range(len(rpn_out["boxes"])):
            rpn_out_transpose.append(
                {"boxes": proposals[i], "scores": objectness_scores[i]}
            )

        ## will rescale boxes to original size
        proposals_out = self.transform.postprocess(
            rpn_out_transpose, images.image_sizes, original_image_sizes
        )

        # detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        roi_heads_out = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections, detector_losses, box_features = itemgetter(
            "result", "losses", "box_features"
        )(roi_heads_out)

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        box_features = box_features.split(boxes_per_image, 0)
        for elt, features in zip(proposals_out, box_features):
            elt["features"] = features

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return {"losses": losses, "detections": detections, "proposals": proposals_out}


import torch.nn as nn


def filter_ret(
    ret, iou_threshold=1.0 / 2, max_proposals=300, min_objectness_score=0.05
):
    score_mask = ret["scores"] > min_objectness_score
    ret = {k: v[score_mask] for (k, v) in ret.items()}

    dedup_idxs = box_ops.batched_nms(
        ret["boxes"],
        scores=ret["scores"],
        idxs=torch.zeros(ret["scores"].shape[0]),
        iou_threshold=iou_threshold,
    )

    top_idxs = dedup_idxs[:max_proposals]
    dedup_ret = {k: v[top_idxs] for (k, v) in ret.items()}
    return dedup_ret


class AgnosticRoIExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = XGeneralizedRCNN(model)

    def forward(self, images: ImageList):
        d = self.model(images)
        rs = []
        for r in d["proposals"]:
            frs = filter_ret(r)
            rs.append(frs)
        return rs


import pandas as pd


def to_dataframe(pairs):
    from ray.data.extensions import TensorArray

    def to_numpy(d):
        return {k: v.detach().cpu().numpy() for (k, v) in d.items()}

    def box2dict(boxes):
        return {
            "x1": boxes[:, 0],
            "y1": boxes[:, 1],
            "x2": boxes[:, 2],
            "y2": boxes[:, 3],
        }

    def paddedBox2Dict(boxes): 
        return {
            "_x1": boxes[:, 0],
            "_y1": boxes[:, 1],
            "_x2": boxes[:, 2],
            "_y2": boxes[:, 3],
        }

    dfs = []
    for (filename, d) in pairs:
        d2 = to_numpy(d)
        rdf = None
        if "new_boxes" in d2.keys(): 
            rdf = pd.DataFrame.from_dict(
                {
                    "filename": filename,
                    **box2dict(d2["boxes"]),
                    **paddedBox2Dict(d2["new_boxes"]),
                    "object_score": d2["scores"],
                    "features": TensorArray(d2["features"]),
                    "video_id": filename.split('/')[1], 
                    "track_id": d2["track_id"],
                }
            )
        else: 
            rdf = pd.DataFrame.from_dict(
                {
                    "filename": filename,
                    **box2dict(d2["boxes"]),
                    "object_score": d2["scores"],
                    "features": TensorArray(d2["features"]),
                    "video_id": d2["video_id"], 
                    "track_id": d2["track_id"],
                }
            )
        dfs.append(rdf)

    return pd.concat(dfs, ignore_index=True)
