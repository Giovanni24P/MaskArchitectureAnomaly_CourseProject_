# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import csv
import glob
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, roc_curve
from torchvision.transforms import Compose, Resize


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


REPO_ROOT = Path(__file__).resolve().parents[1]
EOMT_ROOT = REPO_ROOT / "eomt"
if str(EOMT_ROOT) not in sys.path:
    sys.path.insert(0, str(EOMT_ROOT))


target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def fpr_at_95_tpr(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, 0.95, side="left")
    if idx >= len(fpr):
        return 1.0
    return fpr[idx]


def parse_size(value, default):
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, str):
        parts = value.replace("x", ",").split(",")
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f"Expected H,W size, got {value}")
    return (int(parts[0]), int(parts[1]))


def nested_get(data, keys, default=None):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def infer_defaults(config):
    data_class = nested_get(config, ["data", "class_path"], "")
    if "cityscapes" in data_class.lower():
        return 19, (1024, 1024)
    if "coco" in data_class.lower():
        return 133, (640, 640)
    return 19, (1024, 1024)


def load_eomt(config_path, ckpt_path, device, img_size=None, num_classes=None):
    try:
        from models.eomt import EoMT
        from models.vit import ViT
    except ModuleNotFoundError as exc:
        missing = exc.name or "dependency"
        raise SystemExit(
            f"Missing EoMT dependency '{missing}'. Install eomt/requirements.txt "
            "before running this evaluator."
        ) from exc

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    default_classes, default_img_size = infer_defaults(config)
    img_size = parse_size(
        img_size,
        parse_size(
            nested_get(config, ["data", "init_args", "img_size"]),
            default_img_size,
        ),
    )
    num_classes = int(
        num_classes
        or nested_get(config, ["data", "init_args", "num_classes"], default_classes)
    )

    network_args = nested_get(config, ["model", "init_args", "network", "init_args"], {})
    encoder_args = nested_get(network_args, ["encoder", "init_args"], {})

    encoder = ViT(
        img_size=img_size,
        patch_size=encoder_args.get("patch_size", 16),
        backbone_name=encoder_args.get("backbone_name", "vit_base_patch14_reg4_dinov2"),
        ckpt_path=ckpt_path,
    )
    model = EoMT(
        encoder=encoder,
        num_classes=num_classes,
        num_q=int(network_args.get("num_q", 200)),
        num_blocks=int(network_args.get("num_blocks", 3)),
        masked_attn_enabled=bool(network_args.get("masked_attn_enabled", True)),
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k.replace("._orig_mod", ""): v for k, v in state.items()}
    if any(k.startswith("network.") for k in state):
        state = {k.removeprefix("network."): v for k, v in state.items()}

    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        print(f"Warning: ignored unexpected checkpoint keys: {incompatible.unexpected_keys}")
    if incompatible.missing_keys:
        print(f"Warning: missing checkpoint keys: {incompatible.missing_keys}")

    return model.to(device).eval(), img_size, num_classes


def scale_img_size(size, model_img_size):
    factor = max(
        model_img_size[0] / size[0],
        model_img_size[1] / size[1],
    )
    return [round(s * factor) for s in size]


def window_img(img, model_img_size):
    new_h, new_w = scale_img_size(img.shape[-2:], model_img_size)
    pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    resized = torch.from_numpy(np.array(resized)).permute(2, 0, 1).to(img.device)

    crops, origins = [], []
    num_crops = math.ceil(max(resized.shape[-2:]) / min(model_img_size))
    overlap = num_crops * min(model_img_size) - max(resized.shape[-2:])
    overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

    for j in range(num_crops):
        start = int(j * (min(model_img_size) - overlap_per_crop))
        end = start + min(model_img_size)
        if resized.shape[-2] > resized.shape[-1]:
            crop = resized[:, start:end, :]
        else:
            crop = resized[:, :, start:end]
        crops.append(crop)
        origins.append((start, end))

    return torch.stack(crops), origins


def revert_window_scores(crop_scores, origins, img_size, model_img_size):
    scaled_h, scaled_w = scale_img_size(img_size, model_img_size)
    sums = torch.zeros(
        (crop_scores.shape[1], scaled_h, scaled_w), device=crop_scores.device
    )
    counts = torch.zeros_like(sums)

    for crop_i, (start, end) in enumerate(origins):
        if img_size[0] > img_size[1]:
            sums[:, start:end, :] += crop_scores[crop_i]
            counts[:, start:end, :] += 1
        else:
            sums[:, :, start:end] += crop_scores[crop_i]
            counts[:, :, start:end] += 1

    return F.interpolate(
        (sums / counts.clamp_min(1))[None, ...],
        img_size,
        mode="bilinear",
        align_corners=False,
    )[0]


def eomt_pixel_scores(model, img, model_img_size):
    crops, origins = window_img(img, model_img_size)
    crop_scores = []

    for crop in crops:
        mask_layers, class_layers = model((crop[None].float() / 255.0))
        mask_logits = F.interpolate(
            mask_layers[-1],
            model_img_size,
            mode="bilinear",
            align_corners=False,
        )
        class_logits = class_layers[-1][..., :-1]
        pixel_scores = torch.einsum(
            "bqhw,bqc->bchw",
            mask_logits.sigmoid(),
            class_logits,
        )
        crop_scores.append(pixel_scores[0])

    return revert_window_scores(
        torch.stack(crop_scores),
        origins,
        img.shape[-2:],
        model_img_size,
    )


def anomaly_maps(pixel_scores, methods):
    probs = torch.softmax(pixel_scores, dim=0)
    scores = {}
    if "msp" in methods:
        scores["msp"] = 1.0 - probs.max(dim=0).values
    if "maxlogit" in methods:
        scores["maxlogit"] = -pixel_scores.max(dim=0).values
    if "maxentropy" in methods:
        scores["maxentropy"] = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=0)
    if "rba" in methods:
        scores["rba"] = -pixel_scores.tanh().sum(dim=0)
    return {name: value.detach().cpu().numpy() for name, value in scores.items()}


def gt_path_for_image(path):
    path_gt = path.replace("images", "labels_masks")
    if "RoadObsticle21" in path_gt:
        path_gt = path_gt.replace("webp", "png")
    if "fs_static" in path_gt:
        path_gt = path_gt.replace("jpg", "png")
    if "RoadAnomaly" in path_gt:
        path_gt = path_gt.replace("jpg", "png")
    return path_gt


def load_ood_gt(path_gt):
    mask = target_transform(Image.open(path_gt))
    ood_gts = np.array(mask)

    if "RoadAnomaly" in path_gt:
        ood_gts = np.where((ood_gts == 2), 1, ood_gts)
    if "LostAndFound" in path_gt:
        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
        ood_gts = np.where((ood_gts == 1), 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
    if "Streethazard" in path_gt:
        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
        ood_gts = np.where((ood_gts < 20), 0, ood_gts)
        ood_gts = np.where((ood_gts == 255), 1, ood_gts)

    return ood_gts


def expand_inputs(patterns):
    input_paths = []
    for pattern in patterns:
        input_paths.extend(glob.glob(os.path.expanduser(str(pattern))))
    return sorted(input_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Input images, or glob patterns such as 'RoadAnomaly21/images/*.jpg'.",
    )
    parser.add_argument("--config", required=True, help="EoMT yaml config path.")
    parser.add_argument("--checkpoint", required=True, help="EoMT checkpoint path.")
    parser.add_argument("--model-name", default="EoMT")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--img-size", nargs=2, type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--method",
        default="all",
        choices=["msp", "maxlogit", "maxentropy", "rba", "all"],
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[1.0],
        help="Temperature values for calibration experiments.",
    )
    parser.add_argument("--results-file", default="results_eomt.csv")
    args = parser.parse_args()

    methods = (
        ["msp", "maxlogit", "maxentropy", "rba"]
        if args.method == "all"
        else [args.method]
    )
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print(f"Loading EoMT config: {args.config}")
    print(f"Loading EoMT checkpoint: {args.checkpoint}")
    model, model_img_size, _ = load_eomt(
        args.config,
        args.checkpoint,
        device,
        img_size=args.img_size,
        num_classes=args.num_classes,
    )
    print(f"Model loaded with img_size={model_img_size}")

    input_paths = expand_inputs(args.input)
    if not input_paths:
        raise SystemExit("No input images matched.")

    rows = []
    score_lists_by_temp = {
        temperature: {method: [] for method in methods}
        for temperature in args.temperatures
    }
    gt_list = []

    for path in input_paths:
        print(path)
        image = Image.open(path).convert("RGB").resize((1024, 512), Image.BILINEAR)
        img = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(device)

        with torch.no_grad():
            pixel_scores = eomt_pixel_scores(model, img, model_img_size)

        ood_gts = load_ood_gt(gt_path_for_image(path))
        if 1 not in np.unique(ood_gts):
            continue

        gt_list.append(ood_gts)
        for temperature in args.temperatures:
            score_maps = anomaly_maps(pixel_scores / temperature, methods)
            for method, score_map in score_maps.items():
                score_lists_by_temp[temperature][method].append(score_map)

        del pixel_scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not gt_list:
        raise SystemExit("No images with OOD pixels found.")

    ood_gts = np.array(gt_list)
    ood_mask = ood_gts == 1
    ind_mask = ood_gts == 0

    for temperature in args.temperatures:
        score_lists = score_lists_by_temp[temperature]

        for method in methods:
            anomaly_scores = np.array(score_lists[method])
            ood_out = anomaly_scores[ood_mask]
            ind_out = anomaly_scores[ind_mask]

            labels = np.concatenate((np.zeros(len(ind_out)), np.ones(len(ood_out))))
            values = np.concatenate((ind_out, ood_out))

            auprc = average_precision_score(labels, values) * 100.0
            fpr95 = fpr_at_95_tpr(values, labels) * 100.0
            rows.append(
                {
                    "model": args.model_name,
                    "dataset": args.dataset_name,
                    "temperature": temperature,
                    "method": method,
                    "auprc": auprc,
                    "fpr95": fpr95,
                }
            )
            print(
                f"T={temperature} {method}: "
                f"AUPRC={auprc:.4f} FPR@TPR95={fpr95:.4f}"
            )

    if rows:
        write_header = not os.path.exists(args.results_file)
        with open(args.results_file, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        print(f"Saved results to {args.results_file}")


if __name__ == "__main__":
    main()
