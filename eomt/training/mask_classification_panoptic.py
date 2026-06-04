# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.coco_panoptic import CLASS_MAPPING
from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule

CITYSCAPES_IGNORE_IDX = 255
CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_UNMAPPED_PRED_IDX = 19
CITYSCAPES_METRIC_NUM_CLASSES = 20

# COCO panoptic category id -> Cityscapes train id.
# Classes without a reasonable semantic equivalent become an extra prediction
# class so they count as mistakes on Cityscapes pixels instead of being ignored.
COCO_TO_CITYSCAPES_BY_CATEGORY_ID = {
    1: 11,  # person -> person
    2: 18,  # bicycle -> bicycle
    3: 13,  # car -> car
    4: 17,  # motorcycle -> motorcycle
    6: 15,  # bus -> bus
    7: 16,  # train -> train
    8: 14,  # truck -> truck
    10: 6,  # traffic light -> traffic light
    13: 7,  # stop sign -> traffic sign
    119: 8,  # flower -> vegetation
    125: 9,  # gravel -> terrain
    128: 2,  # house -> building
    149: 0,  # road -> road
    154: 9,  # sand -> terrain
    171: 3,  # wall-brick -> wall
    175: 3,  # wall-stone -> wall
    176: 3,  # wall-tile -> wall
    177: 3,  # wall-wood -> wall
    184: 8,  # tree-merged -> vegetation
    185: 4,  # fence-merged -> fence
    187: 10,  # sky-other-merged -> sky
    191: 1,  # pavement-merged -> sidewalk
    192: 9,  # mountain-merged -> terrain
    193: 8,  # grass-merged -> vegetation
    194: 9,  # dirt-merged -> terrain
    197: 2,  # building-other-merged -> building
    198: 9,  # rock-merged -> terrain
    199: 3,  # wall-other-merged -> wall
}


def _coco_index_to_cityscapes_train_id(num_classes: int) -> torch.Tensor:
    mapping = torch.full(
        (num_classes + 1,), CITYSCAPES_UNMAPPED_PRED_IDX, dtype=torch.long
    )
    for coco_category_id, cityscapes_train_id in COCO_TO_CITYSCAPES_BY_CATEGORY_ID.items():
        coco_idx = CLASS_MAPPING.get(coco_category_id)
        if coco_idx is not None and coco_idx < num_classes:
            mapping[coco_idx] = cityscapes_train_id
    return mapping


class MaskClassificationPanoptic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        stuff_classes: list[int],
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = stuff_classes
        self.coco_to_cityscapes_semantic_eval = False
        self.register_buffer(
            "coco_to_cityscapes",
            _coco_index_to_cityscapes_train_id(num_classes),
            persistent=False,
        )

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        thing_classes = [i for i in range(num_classes) if i not in stuff_classes]
        self.init_metrics_panoptic(
            thing_classes,
            stuff_classes,
            self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1,
        )

    def _enable_coco_to_cityscapes_semantic_eval(self):
        if self.coco_to_cityscapes_semantic_eval:
            return

        self.coco_to_cityscapes_semantic_eval = True
        self.init_metrics_semantic(
            CITYSCAPES_IGNORE_IDX,
            self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1,
            num_classes=CITYSCAPES_METRIC_NUM_CLASSES,
        )
        self.metrics.to(self.device)

    def _on_eval_epoch_end_coco_to_cityscapes_semantic(self, log_prefix):
        for i, metric in enumerate(self.metrics):  # type: ignore
            iou_per_class = metric.compute()
            metric.reset()

            block_postfix = self.block_postfix(i)
            cityscapes_iou = iou_per_class[:CITYSCAPES_NUM_CLASSES]
            for class_idx, iou in enumerate(cityscapes_iou):
                self.log(
                    f"metrics/{log_prefix}_iou_class_{class_idx}{block_postfix}",
                    iou,
                )

            self.log(
                f"metrics/{log_prefix}_iou_all{block_postfix}",
                float(cityscapes_iou.mean()),
            )
            self._log_cityscapes_common_iou(
                cityscapes_iou, log_prefix, block_postfix
            )

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch
        coco_to_cityscapes_semantic_eval = any(
            target.get("eval_task") == "coco_to_cityscapes_semantic"
            for target in targets
        )
        if coco_to_cityscapes_semantic_eval:
            self._enable_coco_to_cityscapes_semantic_eval()

        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(transformed_imgs)

        if coco_to_cityscapes_semantic_eval:
            targets = self.to_per_pixel_targets_semantic(
                targets, CITYSCAPES_IGNORE_IDX
            )
        else:
            is_crowds = [target["is_crowd"] for target in targets]
            targets = self.to_per_pixel_targets_panoptic(targets)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, img_sizes
            )
            preds = self.to_per_pixel_preds_panoptic(
                mask_logits,
                class_logits,
                self.stuff_classes,
                self.mask_thresh,
                self.overlap_thresh,
            )
            if coco_to_cityscapes_semantic_eval:
                preds = [
                    self.coco_to_cityscapes[pred[:, :, 0].clamp_min(0)]
                    for pred in preds
                ]
                self.update_metrics_semantic(preds, targets, i)
            else:
                self.update_metrics_panoptic(preds, targets, is_crowds, i)

    def on_validation_epoch_end(self):
        if self.coco_to_cityscapes_semantic_eval:
            self._on_eval_epoch_end_coco_to_cityscapes_semantic("val")
        else:
            self._on_eval_epoch_end_panoptic("val")

    def on_validation_end(self):
        if self.coco_to_cityscapes_semantic_eval:
            self._on_eval_end_semantic("val")
        else:
            self._on_eval_end_panoptic("val")
