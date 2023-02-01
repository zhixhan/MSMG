from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
import numpy as np

from losses.oim import OIMLoss
from models.resnet import build_resnet
from models.transhead_hat3_3da_sum_share import HATHead

class SeqNet(nn.Module):
    def __init__(self, cfg):
        super(SeqNet, self).__init__()

        backbone, box_head = build_resnet(name="resnet50", pretrained=True)

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOPN_TEST
        )
        post_nms_top_n = dict(
            training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )
        #box_head = TransformerHead(feature_names=cfg.MODEL.ROI_HEAD.TRANSFORMERHEAD_FEATNAME, in_channels=cfg.MODEL.ROI_HEAD.TRANSFORMERHEAD_INCHANNELS, depth=cfg.MODEL.ROI_HEAD.TRANSFORMERHEAD_DEPTH, embed_dim=cfg.MODEL.ROI_HEAD.TRANSFORMERHEAD_DIM, spacial_size=cfg.MODEL.ROI_HEAD.TRANSFORMERHEAD_SPACIAL_SIZE)
        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        reid_head = HATHead(feature_names=cfg.MODEL.ROI_HEAD.TRANSFORMERHEAD_FEATNAME, in_channels=cfg.MODEL.ROI_HEAD.TRANSFORMERHEAD_INCHANNELS, depth=cfg.MODEL.ROI_HEAD.TRANSFORMERHEAD_DEPTH, embed_dim=cfg.MODEL.ROI_HEAD.TRANSFORMERHEAD_DIM, spacial_size=cfg.MODEL.ROI_HEAD.TRANSFORMERHEAD_SPACIAL_SIZE)
        box_roi_pool = nn.ModuleDict([
            [fname, MultiScaleRoIAlign(
                featmap_names=[fname], output_size=cfg.MODEL.ROI_HEAD.TRANSFORMERHEAD_SPACIAL_SIZE[j], sampling_ratio=2
            )]
            for j, fname in enumerate(cfg.MODEL.ROI_HEAD.TRANSFORMERHEAD_FEATNAME)
        ])
        box_predictor = BBoxRegressor(2048, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)
        embedding_head = NormAwareEmbedding(featmap_names=cfg.MODEL.ROI_HEAD.NAE_FEATNAME, in_channels=cfg.MODEL.ROI_HEAD.TRANSFORMERHEAD_OUTPUT_DIM, indv_dims=cfg.MODEL.ROI_HEAD.NAE_INDV_DIM, dim=np.sum(cfg.MODEL.ROI_HEAD.NAE_INDV_DIM))
        #embedding_head = NormAwareEmbedding(featmap_names=['trans_feat_res4'], in_channels=[2048])
        self.inference_last_embed = 3
        roi_heads = SeqRoIHeads(
            # OIM
            reid_loss=HardOIMLoss if cfg.MODEL.LOSS.HARD_MINING else OIMLoss,
            num_pids=cfg.MODEL.LOSS.LUT_SIZE,
            num_cq_size=cfg.MODEL.LOSS.CQ_SIZE,
            oim_momentum=cfg.MODEL.LOSS.OIM_MOMENTUM,
            oim_scalar=cfg.MODEL.LOSS.OIM_SCALAR,
            max_epoch=cfg.SOLVER.MAX_EPOCHS - 1,
            # SeqNet
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            embedding_head=embedding_head,
            inference_last_embed=self.inference_last_embed,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
            oim_dim=np.sum(cfg.MODEL.ROI_HEAD.NAE_INDV_DIM)
        )

        transform = GeneralizedRCNNTransform(
            min_size=cfg.INPUT.MIN_SIZE,
            max_size=cfg.INPUT.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform
        

        # loss weights
        self.lw_rpn_reg = cfg.SOLVER.LW_RPN_REG
        self.lw_rpn_cls = cfg.SOLVER.LW_RPN_CLS
        self.lw_proposal_reg = cfg.SOLVER.LW_PROPOSAL_REG
        self.lw_proposal_cls = cfg.SOLVER.LW_PROPOSAL_CLS
        self.lw_box_reg = cfg.SOLVER.LW_BOX_REG
        self.lw_box_cls = cfg.SOLVER.LW_BOX_CLS
        self.lw_box_reid_1st = cfg.SOLVER.LW_BOX_REID_1ST
        self.lw_box_reid_2nd = cfg.SOLVER.LW_BOX_REID_2ND
        self.lw_box_reid_3rd = cfg.SOLVER.LW_BOX_REID_3RD

    def inference(self, images, targets=None, query_img_as_gallery=False):
        """
        query_img_as_gallery: Set to True to detect all people in the query image.
            Meanwhile, the gt box should be the first of the detected boxes.
            This option serves CBGM.
        """
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        if query_img_as_gallery:
            assert targets is not None

        if targets is not None and not query_img_as_gallery:
            # query
            boxes = [t["boxes"] for t in targets]
            box_features = OrderedDict()
            for fname, box_roi_pooling in self.roi_heads.box_roi_pool.items():
                box_features[fname] = box_roi_pooling(features, boxes, images.image_sizes)
            box_features_1st, box_features_2nd, box_features_3rd = self.roi_heads.reid_head(box_features)
            embeddings_1st = self.roi_heads.embedding_head_1st(box_features_1st)
            embeddings_2nd = self.roi_heads.embedding_head_2nd(box_features_2nd)
            embeddings_3rd = self.roi_heads.embedding_head_3rd(box_features_3rd)
            if self.inference_last_embed == 1:
                embeddings = embeddings_3rd
            elif self.inference_last_embed == 2:
                embeddings = torch.cat([embeddings_2nd, embeddings_3rd], dim=-1)
            else:
                embeddings = torch.cat([embeddings_1st, embeddings_2nd, embeddings_3rd], dim=-1)
            return embeddings.split(1, 0)
        else:
            # gallery
            proposals, _ = self.rpn(images, OrderedDict([["feat_res4", features["feat_res4"]]]), targets)
            detections, _, _, _, _, _ = self.roi_heads(
                features, proposals, images.image_sizes, targets, query_img_as_gallery
            )
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            return detections

    def forward(self, images, targets=None, query_img_as_gallery=False, epoch=None):
        if not self.training:
            return self.inference(images, targets, query_img_as_gallery)
        
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, OrderedDict([["feat_res4", features["feat_res4"]]]), targets)
        _, detector_losses, feats_reid_1st, feats_reid_2nd, feats_reid_3rd, targets_reid = self.roi_heads(features, proposals, images.image_sizes, targets, epoch=epoch)

        # rename rpn losses to be consistent with detection losses
        proposal_losses["loss_rpn_reg"] = proposal_losses.pop("loss_rpn_box_reg")
        proposal_losses["loss_rpn_cls"] = proposal_losses.pop("loss_objectness")

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # apply loss weights
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_proposal_reg"] *= self.lw_proposal_reg
        losses["loss_proposal_cls"] *= self.lw_proposal_cls
        losses["loss_rcnn_reid_1st"] *= self.lw_box_reid_1st
        losses["loss_rcnn_reid_2nd"] *= self.lw_box_reid_2nd
        losses["loss_rcnn_reid_3rd"] *= self.lw_box_reid_3rd
        return losses, feats_reid_1st, feats_reid_2nd, feats_reid_3rd, targets_reid


class SeqRoIHeads(RoIHeads):
    def __init__(
        self,
        reid_loss,
        num_pids,
        num_cq_size,
        oim_momentum,
        oim_scalar,
        faster_rcnn_predictor,
        reid_head,
        embedding_head,
        inference_last_embed,
        max_epoch=None,
        oim_dim=256,
        *args,
        **kwargs
    ):
        super(SeqRoIHeads, self).__init__(*args, **kwargs)
        self.embedding_head_1st = embedding_head
        self.embedding_head_2nd = deepcopy(self.embedding_head_1st)
        self.embedding_head_3rd = deepcopy(self.embedding_head_1st)
        self.oim_dim = oim_dim
        self.reid_loss_1st = reid_loss(oim_dim, num_pids, num_cq_size, oim_momentum, oim_scalar, max_epoch = max_epoch)
        self.reid_loss_2nd = deepcopy(self.reid_loss_1st)
        self.reid_loss_3rd = deepcopy(self.reid_loss_1st)
        self.faster_rcnn_predictor = faster_rcnn_predictor
        self.reid_head = reid_head
        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections
        self.inference_last_embed = inference_last_embed
        
       

    def forward(self, features, proposals, image_shapes, targets=None, query_img_as_gallery=False, epoch=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if self.training:
            proposals, _, proposal_pid_labels, proposal_reg_targets, proposals_ious = self.select_training_samples(
                proposals, targets
            )

        # ------------------- Faster R-CNN head ------------------ #
        
        proposal_features = self.box_roi_pool["feat_res4"](OrderedDict([["feat_res4", features["feat_res4"]]]), proposals, image_shapes)
        
        proposal_features,_ = self.box_head(proposal_features)
        proposal_cls_scores, proposal_regs = self.faster_rcnn_predictor(
            proposal_features["feat_res5"]
        )

        if self.training:
            boxes = self.get_boxes(proposal_regs, proposals, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            boxes, _, box_pid_labels, box_reg_targets, boxes_ious = self.select_training_samples(boxes, targets)
        else:
            # invoke the postprocess method inherited from parent class to process proposals
            boxes, scores, _ = self.postprocess_proposals(
                proposal_cls_scores, proposal_regs, proposals, image_shapes
            )

        cws = True
        gt_det = None
        if not self.training and query_img_as_gallery:
            # When regarding the query image as gallery, GT boxes may be excluded
            # from detected boxes. To avoid this, we compulsorily include GT in the
            # detection results. Additionally, CWS should be disabled as the
            # confidences of these people in query image are 1
            cws = False
            gt_box = [targets[0]["boxes"]]
            gt_box_features = OrderedDict()
            for fname, box_roi_pooling in self.box_roi_pool.items():
                gt_box_features[fname] = box_roi_pooling(features, gt_box, image_shapes)
            gt_box_features_1st, gt_box_features_2nd, gt_box_features_3rd = self.reid_head(gt_box_features)
            embeddings_1st = self.embedding_head_1st(gt_box_features_1st)
            embeddings_2nd = self.embedding_head_2nd(gt_box_features_2nd)
            embeddings_3rd = self.embedding_head_3rd(gt_box_features_3rd)
            if self.inference_last_embed == 1:
                embeddings = embeddings_3rd
            elif self.inference_last_embed == 2:
                embeddings = torch.cat([embeddings_2nd, embeddings_3rd], dim=-1)
            else:
                embeddings = torch.cat([embeddings_1st, embeddings_2nd, embeddings_3rd], dim=-1)
            gt_det = {"boxes": targets[0]["boxes"], "embeddings": embeddings}
        '''
        if not self.training:
            result = []
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i], scores=scores[i]
                    )
                )
            return result, None, None, None
        '''
        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0:
            assert not self.training
            boxes = gt_det["boxes"] if gt_det else torch.zeros(0, 4)
            labels = torch.ones(1).type_as(boxes) if gt_det else torch.zeros(0)
            scores = torch.ones(1).type_as(boxes) if gt_det else torch.zeros(0)
            embeddings = gt_det["embeddings"] if gt_det else torch.zeros(0, self.oim_dim)
            return [dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)], [], None, None, None, None

        # --------------------- Baseline head -------------------- #
        box_features = OrderedDict()
        for fname, box_roi_pooling in self.box_roi_pool.items():
            box_features[fname] = box_roi_pooling(features, boxes, image_shapes)
        box_features_1st, box_features_2nd, box_features_3rd = self.reid_head(box_features)
       
        box_embeddings_1st = self.embedding_head_1st(box_features_1st)
        box_embeddings_2nd = self.embedding_head_2nd(box_features_2nd)
        box_embeddings_3rd = self.embedding_head_3rd(box_features_3rd)

        result, losses = [], {}
        feats_reid_1st, feats_reid_2nd, feats_reid_3rd, targets_reid = None, None, None, None
        if self.training:
            proposal_labels = [y.clamp(0, 1) for y in proposal_pid_labels]
            losses = detection_losses(
                proposal_cls_scores,
                proposal_regs,
                proposal_labels,
                proposal_reg_targets,
            )
            loss_rcnn_reid_1st, feats_reid_1st, targets_reid = self.reid_loss_1st(box_embeddings_1st, box_pid_labels, boxes_ious, epoch=epoch)
            loss_rcnn_reid_2nd, feats_reid_2nd, _ = self.reid_loss_2nd(box_embeddings_2nd, box_pid_labels, boxes_ious, epoch=epoch)
            loss_rcnn_reid_3rd, feats_reid_3rd, _ = self.reid_loss_3rd(box_embeddings_3rd, box_pid_labels, boxes_ious, epoch=epoch)
            losses.update(loss_rcnn_reid_1st=loss_rcnn_reid_1st)
            losses.update(loss_rcnn_reid_2nd=loss_rcnn_reid_2nd)
            losses.update(loss_rcnn_reid_3rd=loss_rcnn_reid_3rd)
            #loss_box_reid, feats_reid, targets_reid = self.reid_loss(box_embeddings, box_pid_labels, boxes_ious, epoch=epoch)
            #losses.update(loss_box_reid=loss_box_reid)
        else:
            # The IoUs of these boxes are higher than that of proposals,
            # so a higher NMS threshold is needed
            orig_thresh = self.nms_thresh
            self.nms_thresh = 0.5
            #self.score_thresh = 0.0
            if self.inference_last_embed == 1:
                box_embeddings = box_embeddings_3rd
            elif self.inference_last_embed == 2:
                box_embeddings = torch.cat([box_embeddings_2nd, box_embeddings_3rd], dim=-1)
            else:
                box_embeddings = torch.cat([box_embeddings_1st, box_embeddings_2nd, box_embeddings_3rd], dim=-1)
            boxes, scores, embeddings, labels = self.postprocess_boxes(
                box_embeddings,
                boxes,
                image_shapes,
                fcs=scores,
                gt_det=gt_det,
                cws=cws,
            )
            # set to original thresh after finishing postprocess
            self.nms_thresh = orig_thresh
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i], labels=labels[i], scores=scores[i], embeddings=embeddings[i]
                    )
                )
        return result, losses, feats_reid_1st, feats_reid_2nd, feats_reid_3rd, targets_reid

    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
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
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            #print(boxes, image_shapes)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            #print(boxes)
            #print('=================================')

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

    def get_boxes(self, box_regression, proposals, image_shapes):
        """
        Get boxes from proposals.
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes = pred_boxes.split(boxes_per_image, 0)

        all_boxes = []
        for boxes, image_shape in zip(pred_boxes, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # remove predictions with the background label
            boxes = boxes[:, 1:].reshape(-1, 4)
            all_boxes.append(boxes)

        return all_boxes

    def postprocess_boxes(
        self,
        embeddings,
        proposals,
        image_shapes,
        fcs=None,
        gt_det=None,
        cws=True,
    ):
        """
        Similar to RoIHeads.postprocess_detections, but can handle embeddings and implement
        First Classification Score (FCS).
        """
        device = embeddings.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = proposals

        
        pred_scores = fcs[0]
       
        if cws:
            # Confidence Weighted Similarity (CWS)
            embeddings = embeddings * pred_scores.view(-1, 1)

        # split boxes and scores per image
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings.split(boxes_per_image, 0)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(
            pred_boxes, pred_scores, pred_embeddings, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes.unsqueeze(1)
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            if self.inference_last_embed == 1:
                dim = self.embedding_head_3rd.dim 
            elif self.inference_last_embed == 2:
                dim = self.embedding_head_3rd.dim * 2
            else:
                dim = self.embedding_head_3rd.dim * 3 
            embeddings = embeddings.reshape(-1, dim)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            
            boxes, scores, labels, embeddings = (
                boxes[inds],
                scores[inds],
                labels[inds],
                embeddings[inds],
            )

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )
            
            if gt_det is not None:
                # include GT into the detection results
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                embeddings = torch.cat((embeddings, gt_det["embeddings"]), dim=0)

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels

    def select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
        ious = []
        for proposal, matched_gt_box in zip(proposals, matched_gt_boxes):
            ious.append(torch.diag(box_ops.box_iou(proposal, matched_gt_box)))
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, ious

class NormAwareEmbedding(nn.Module):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.
    """

    def __init__(self, featmap_names=["trans_feat_res4"], in_channels=[2048], dim=256, indv_dims=None):
        super(NormAwareEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim

        self.projectors = nn.ModuleDict()
        if indv_dims is None:
            indv_dims = self._split_embedding_dim()
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(nn.Linear(in_channel, indv_dim), nn.BatchNorm1d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        assert len(featmaps) == len(self.featmap_names)
        if len(featmaps) == 1:
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                embeddings = self.projectors[k](v)
                norms = embeddings.norm(2, 1, keepdim=True)
                embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
                return embeddings
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(self.projectors[k](v))
            embeddings = torch.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            return embeddings

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim // parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp


class BBoxRegressor(nn.Module):
    """
    Bounding box regression layer.
    """

    def __init__(self, in_channels, num_classes=2, bn_neck=True):
        """
        Args:
            in_channels (int): Input channels.
            num_classes (int, optional): Defaults to 2 (background and pedestrian).
            bn_neck (bool, optional): Whether to use BN after Linear. Defaults to True.
        """
        super(BBoxRegressor, self).__init__()
        if bn_neck:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes), nn.BatchNorm1d(4 * num_classes)
            )
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, 4 * num_classes)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas


def detection_losses(
    proposal_cls_scores,
    proposal_regs,
    proposal_labels,
    proposal_reg_targets
):
    proposal_labels = torch.cat(proposal_labels, dim=0)
    proposal_reg_targets = torch.cat(proposal_reg_targets, dim=0)

    loss_proposal_cls = F.cross_entropy(proposal_cls_scores, proposal_labels)

    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
    sampled_pos_inds_subset = torch.nonzero(proposal_labels > 0).squeeze(1)
    labels_pos = proposal_labels[sampled_pos_inds_subset]
    N = proposal_cls_scores.size(0)
    proposal_regs = proposal_regs.reshape(N, -1, 4)

    loss_proposal_reg = F.smooth_l1_loss(
        proposal_regs[sampled_pos_inds_subset, labels_pos],
        proposal_reg_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_proposal_reg = loss_proposal_reg / proposal_labels.numel()

    

    return dict(
        loss_proposal_cls=loss_proposal_cls,
        loss_proposal_reg=loss_proposal_reg,
    )