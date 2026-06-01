# Copyright (c) OpenMMLab. All rights reserved.
import os
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

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


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument(
        '--method',
        default='maxlogit',
        choices=['msp', 'maxlogit', 'maxentropy', 'all'],
        help='Post-hoc anomaly score to evaluate. Use "all" to run every baseline.',
    )
    args = parser.parse_args()
    methods = ['msp', 'maxlogit', 'maxentropy'] if args.method == 'all' else [args.method]
    anomaly_score_lists = {method: [] for method in methods}
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    model = ERFNet(NUM_CLASSES)

    if device.type == 'cuda':
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=device))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    input_paths = []
    for input_pattern in args.input:
        input_paths.extend(glob.glob(os.path.expanduser(str(input_pattern))))

    for path in input_paths:
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

        score_maps = {}
        if 'msp' in methods:
            score_maps['msp'] = 1.0 - probs.max(dim=1).values.squeeze(0).cpu().numpy()
        if 'maxlogit' in methods:
            score_maps['maxlogit'] = -logits.max(dim=1).values.squeeze(0).cpu().numpy()
        if 'maxentropy' in methods:
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
            score_maps['maxentropy'] = entropy.squeeze(0).cpu().numpy()

        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             for method, score_map in score_maps.items():
                 anomaly_score_lists[method].append(score_map)
        del logits, probs, score_maps, ood_gts, mask
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    for method in methods:
        anomaly_scores = np.array(anomaly_score_lists[method])
        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        print(f'{method} AUPRC score: {prc_auc*100.0}')
        print(f'{method} FPR@TPR95: {fpr*100.0}')

        file.write(
            (
                '    Method: ' + method
                + '    AUPRC score:' + str(prc_auc*100.0)
                + '   FPR@TPR95:' + str(fpr*100.0)
            )
        )
    file.close()

if __name__ == '__main__':
    main()
