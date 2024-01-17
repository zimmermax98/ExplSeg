import json
import os, shutil
from collections import Counter

import argh
import gin
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pylab as pylab
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from scipy.ndimage import distance_transform_bf

from segmentation import train
from tqdm import tqdm
from segmentation.dataset import resize_label
from segmentation.constants import CITYSCAPES_CATEGORIES, CITYSCAPES_19_EVAL_CATEGORIES, \
    PASCAL_CATEGORIES, PASCAL_ID_MAPPING
from settings import data_path, log
import cv2
from PIL import Image, ImageDraw as D

import sys
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "ExplSeg", "DeepLabV3"))
import datasets

import sys, os
deeplabv3_path = os.path.join(os.path.expanduser("~"), "ExplSeg", "DeepLabV3")
sys.path.insert(0, deeplabv3_path)
from utils import ext_transforms as et
from datasets.voc import VOCSegmentation

cityscapes_classes = datasets.Cityscapes.classes

train_ids_sorted = np.sort(np.unique([c.train_id for c in cityscapes_classes]))
cityscapes_colors = []
for train_id in train_ids_sorted:
    for i in range(len(cityscapes_classes)):
        if cityscapes_classes[i].train_id == train_id:
            cityscapes_colors.append([*cityscapes_classes[i].color, 255])
            break
cityscapes_colors = np.array(cityscapes_colors) / 255



def run_evaluation(model_name: str, training_phase: str, batch_size: int = 2, pascal: bool = False,
                   margin: int = 0):
    model_path = os.path.join('./results', model_name)
    config_path = os.path.join(model_path, 'config.gin')
    gin.parse_config_file(config_path)

    if training_phase == 'pruned':
        checkpoint_path = os.path.join(model_path, 'pruned/checkpoints/push_last.pth')
    else:
        checkpoint_path = os.path.join(model_path, f'checkpoints/{training_phase}_last.pth')

    log(f'Loading model from {checkpoint_path}')
    ppnet = torch.load(checkpoint_path)  # , map_location=torch.device('cpu'))
    ppnet = ppnet.cuda()
    ppnet.eval()

    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    img_dir = os.path.join(data_path, f'img_with_margin_{margin}/val')

    all_img_files = sorted([p for p in os.listdir(img_dir) if p.endswith('.npy')])

    ann_dir = os.path.join(data_path, 'annotations/val')

    ID_MAPPING = PASCAL_ID_MAPPING if pascal else CITYSCAPES_19_EVAL_CATEGORIES
    CATEGORIES = PASCAL_CATEGORIES if pascal else CITYSCAPES_CATEGORIES

    trainid2id = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
    if pascal:
        id2trainid = {v: k for k, v in ID_MAPPING.items()}
    else:
        id2trainid = ID_MAPPING.copy()
        for i in range(len(id2trainid)):
            if id2trainid[i] == 0:
                id2trainid[i] = 255
            else:
                id2trainid[i] -= 1

    if pascal:
        pred2name = {i: CATEGORIES[k+1] for i, k in trainid2id.items() if k < len(CATEGORIES)-1}
        pred2name_with_void = pred2name.copy()
        pred2name_with_void[255] = "ignore"
    else:
        pred2name = {i: CATEGORIES[k] for i, k in trainid2id.items()}
        pred2name_with_void = pred2name.copy()
        pred2name_with_void[19] = "void"

    cls_prototype_counts = [Counter() for _ in range(len(pred2name))]
    proto_ident = ppnet.prototype_class_identity.cpu().detach().numpy()
    mean_top_k = np.zeros(proto_ident.shape[0], dtype=float)
    
    class_weights = ppnet.last_layer.weight.detach().cpu().numpy()
    n_protos = class_weights.shape[1]

    RESULTS_DIR = os.path.join(model_path, f'evaluation/{training_phase}')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    CLS_CONVERT = np.vectorize(ID_MAPPING.get)

    proto2cls = {}
    cls2protos = {c: [] for c in range(ppnet.num_classes)}

    for proto_num in range(proto_ident.shape[0]):
        cls = np.argmax(proto_ident[proto_num])
        proto2cls[proto_num] = cls
        cls2protos[cls].append(proto_num)

    PROTO2CLS = np.vectorize(proto2cls.get)

    protos = ppnet.prototype_vectors.squeeze()

    all_cls_distances = []

    with torch.no_grad():
        for cls_i in range(ppnet.num_classes):
            cls_proto_ind = (proto_ident[:, cls_i] == 1).nonzero()[0]
            if len(cls_proto_ind) < 2:
                all_cls_distances.append(None)
                continue
            cls_protos = protos[cls_proto_ind]

            distances = torch.cdist(cls_protos, cls_protos)
            distances = distances + 10e6 * torch.triu(torch.ones_like(distances, device=cls_protos.device))
            distances = distances.flatten()
            distances = distances[distances < 10e6]

            distances = distances.cpu().detach().numpy()
            all_cls_distances.append(distances)

    n_rows = 4 if len(pred2name) <= 20 else 5
    fig, axes = plt.subplots(n_rows, 5, figsize=(15, 12))

    plt.suptitle(f'{model_name} ({training_phase})\nHistogram of distances between same-class prototypes')
    axes = axes.flatten()
    class_i = 0

    for class_i, class_name in pred2name.items():
        if all_cls_distances[class_i] is None:
            continue
        axes[class_i].hist(all_cls_distances[class_i], bins=10)
        d_min, d_avg, d_max = np.min(all_cls_distances[class_i]), np.mean(all_cls_distances[class_i]), np.max(
            all_cls_distances[class_i])
        axes[class_i].set_title(f'{class_name}\nmin: {d_min:.2f} avg: {d_avg:.2f} max: {d_max:.2f}')

    for i in range(class_i+1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'histogram_dist_same_class_prototypes.png'))


    # run the following code to visualize on some samples

    N_SAMPLES = 5
    DPI = 100
    np.random.seed(1)
    SORT_PROTOS_BY_CLASS_CONTRIBUTION = True
    N_PROTOS_VIS = 3
    SHOW_ORIG_IMG = True
    SAMPLE_CORRECT = True

    hive_vis_dir = f"explanations_{'correct' if SAMPLE_CORRECT else 'incorrect'}"

    prototype_dir = f"results/{model_name}/prototypes_orig/0_all/"
    if os.path.exists(hive_vis_dir):
        shutil.rmtree(hive_vis_dir)
    os.makedirs(hive_vis_dir)

    #if pascal:
    #    val_dst = VOCSegmentation(root="/fastdata/MT_ExplSeg/datasets/VOC", year="2012",
    #                                image_set='val', download=False, transform=None, unnorm_transform=None)
    #    all_img_files = val_dst.images

    for example_i, img_file in tqdm(enumerate(all_img_files),
                                    total=len(all_img_files), ncols=0, desc='nearest prototype visualization'):
        img = np.load(os.path.join(img_dir, img_file)).astype(np.uint8)

        ann = np.load(os.path.join(ann_dir, img_file))
        if not pascal:
            ann = np.vectorize(id2trainid.get)(ann)

        if pascal:
            ann = resize_label(ann, size=(513, 513)).cpu().detach().numpy()

        if margin != 0:
            img = img[margin:-margin, margin:-margin]
        img_shape = (513, 513) if pascal else (img.shape[0], img.shape[1])

        with torch.no_grad():
            img_tensor = transform(img).unsqueeze(0).cuda()
            img_tensor = torch.nn.functional.interpolate(img_tensor, size=img_shape,
                                                         mode='bilinear', align_corners=False)
            if SORT_PROTOS_BY_CLASS_CONTRIBUTION:
                # call similarties (2nd return value) distances to make life easier
                logits, distances = ppnet.forward(img_tensor, return_activations=True)
                distances = distances.T.unsqueeze(0).view(logits.shape[0], distances.shape[1], logits.shape[1], logits.shape[2])
            else:  # sort protos by DISTANCE
                logits, distances = ppnet.forward(img_tensor)

            img = torch.tensor(img).cuda().permute(2, 0, 1).unsqueeze(0).float()
            img = torch.nn.functional.interpolate(img, size=img_shape,
                                                  mode='bilinear', align_corners=False)
            img = img.cpu().detach().numpy()[0].astype(int)
            img = img.transpose(1, 2, 0)

            logits = logits.permute(0, 3, 1, 2)

            logits = F.interpolate(logits, size=img_shape, mode='bilinear', align_corners=False)[0]
            distances = F.interpolate(distances, size=img_shape, mode='bilinear', align_corners=False)[0]

            # (H, W, C)
            logits = logits.softmax(dim=0).detach().cpu().numpy()  # TODO: Use confidence to select low confidence areas
            distances = distances.cpu().detach().numpy()
            preds = np.argmax(logits, axis=0)
            if pascal:
                #if 1 in np.unique(preds):
                #    print("That shouldnt happen, oops")
                #preds = np.clip(preds-1, a_min=0, a_max=None)
                #preds = np.vectorize(trainid2id.get)(preds)
                pass

            confs = np.max(logits, axis=0)
            conf_threshold = np.percentile(confs, 10)
            conf_mask = (confs > conf_threshold)

            void_mask = (ann == 255)
            not_void_mask = np.invert(void_mask)

        if SORT_PROTOS_BY_CLASS_CONTRIBUTION:
            class_contribution_scores = class_weights[preds].transpose(2, 0, 1) * distances
            nearest_protos = np.argpartition(class_contribution_scores, 
                                            np.arange(n_protos - N_PROTOS_VIS, n_protos), 
                                            axis=0)[::-1][:N_PROTOS_VIS][:N_PROTOS_VIS].transpose(1, 2, 0).squeeze()
        else:  # sort protos by DISTANCE
            nearest_protos = np.argpartition(distances, 
                                            np.arange(n_protos - N_PROTOS_VIS, n_protos), 
                                            axis=0)[::-1][:N_PROTOS_VIS][:N_PROTOS_VIS].transpose(1, 2, 0).squeeze()
        
        
        """
        region_size = 32
        region_y = np.random.randint(img.shape[0] - region_size)
        region_x = np.random.randint(img.shape[1] - region_size)
        region = [region_y, region_y + region_size, region_x, region_x + region_size]
        img_region = img[region[0]:region[1], region[2]:region[3]]

        preds_region = preds[region[0]:region[1], region[2]:region[3]]
        max_pred_region = preds_region.max()
        max_pred_region_mask = preds_region == max_pred_region
        nearest_protos_region = nearest_protos[region[0]:region[1], region[2]:region[3]]
        """
        
        if SAMPLE_CORRECT:
            # correct = 1, incorrect = 0
            correct = (preds == ann)
            
            # set "ignore" pixels to 0, so that distant correct pixels are also far away from "ignore" regions
            correct[ann == 255] = 0

            # ignore background pixels
            if pascal:
                correct[ann == 0] = 0
                
            # pad the image edge with "0" to consider distance from image edge
            pad_width = 1
            correct = np.pad(correct, pad_width, mode="constant", constant_values=0)
            
            # compute the distance for non-zero (correct) pixels 
            # to nearest zero pixels (incorrect pixels, "ignore" pixels or image edge)
            distances2 = distance_transform_bf(correct)
            
            # sample randomly from points with thresholded distance, to avoid pixels on the edge
            max_dist_idxs = np.where(distances2.flatten() > 10)[0]
            if len(max_dist_idxs) == 0:
                continue
            max_dist_idx = np.random.choice(max_dist_idxs)
            max_dist_h_idx, max_dist_w_idx = np.unravel_index(max_dist_idx, correct.shape)
            
            # don't forget to subtract the padding
            max_dist_h_idx -= pad_width
            max_dist_w_idx -= pad_width
        else:
            """
            # correct = 0, incorrect = 1
            incorrect = (preds != ann)
            
            # set "ignore" pixels to 0, so that distant incorrect pixels are also far away from "ignore" regions
            incorrect[ann == 255] = 0
            """

            true_classes = np.unique(ann)
            pred_classes, pred_class_count = np.unique(preds, return_counts=True)

            # find all classes that were predicted, but not in true label
            incorr_pred_classes_mask = np.isin(pred_classes, true_classes, invert=True)
            incorr_pred_classes = pred_classes[incorr_pred_classes_mask]
            incorr_pred_classes_count = pred_class_count[incorr_pred_classes_mask]

            if len(incorr_pred_classes_count) == 0:
                continue
            
            area_threshold = 0.05  # 5%

            argmax = np.argmax(incorr_pred_classes_count)
            if (incorr_pred_classes_count[argmax] / np.prod(ann.shape)) < area_threshold:
                continue
            
            incorrect = (preds == incorr_pred_classes[argmax]) & (ann != 255)
                
            # pad the image edge with "0" to consider distance from image edge
            pad_width = 1
            incorrect = np.pad(incorrect, pad_width, mode="constant", constant_values=0)
            
            # compute the distance for non-zero (incorrect) pixels 
            # to nearest zero pixels (correct pixels, "ignore" pixels or image edge)
            distances2 = distance_transform_bf(incorrect)
            
            # take the points with biggest distance
            max_dist_idx = np.argmax(distances2)
            max_dist_h_idx, max_dist_w_idx = np.unravel_index(max_dist_idx, incorrect.shape)
            
            # don't forget to subtract the padding
            max_dist_h_idx -= pad_width
            max_dist_w_idx -= pad_width

        pixel = [max_dist_h_idx, max_dist_w_idx]
        
        region_size = 32
        #pixel_y = np.random.randint(region_size, img.shape[0] - region_size)
        #pixel_x = np.random.randint(region_size, img.shape[1] - region_size)
        #pixel = [pixel_y, pixel_x]
        img = np.array(transforms.Resize((224, 224))(transforms.ToTensor()(img)).permute(1, 2, 0))
        pixel_y, pixel_x = pixel[0], pixel[1]
        pixel_y_downscale, pixel_x_downscale = round(pixel_y * 224/513), round(pixel_x * 224/513)
        region_h_min = max(0, pixel_y_downscale - region_size)
        region_h_max = min(img.shape[0]-1, pixel_y_downscale + region_size+1)
        region_w_min = max(0, pixel_x_downscale - region_size)
        region_w_max = min(img.shape[1]-1, pixel_x_downscale + region_size+1)
        img_region = img[region_h_min:region_h_max, region_w_min:region_w_max]
        img_region_with_pixel = Image.fromarray(img_region.astype(np.uint8))
        draw = D.Draw(img_region_with_pixel)
        draw.rectangle([(region_size-2, region_size-2), (region_size+2, region_size+2)], outline="red", fill=None, width=1)
        img_region_with_pixel = np.asarray(img_region_with_pixel)
        pred = preds[pixel_y, pixel_x]
        label = ann[pixel_y, pixel_x]
        nearest_protos_pixel = nearest_protos[pixel_y, pixel_x]
        nearest_protos_distances = distances.transpose(1, 2, 0)[pixel_y, pixel_x][nearest_protos_pixel]
        
        img_with_region = Image.fromarray(img.astype(np.uint8))
        draw = D.Draw(img_with_region)
        draw.rectangle([(pixel_x_downscale-region_size-2, pixel_y_downscale-region_size-2), (pixel_x_downscale+region_size+2, pixel_y_downscale+region_size+2)], outline="yellow", fill=None, width=3)
        img_with_region = np.asarray(img_with_region)

        n_rows = N_PROTOS_VIS
        n_cols = 3 if SHOW_ORIG_IMG else 2

        params = {'figure.autolayout': 'True',
                  'figure.titlesize': 'small',
                  'axes.labelsize': 'small',
                  'axes.titlesize': 'small'}
        pylab.rcParams.update(params)

        fig, axs = plt.subplots(n_rows, n_cols,
                                    figsize=(1.6 * 5, (1.6 * n_rows)),
                                    #gridspec_kw={'width_ratios': [4, 4, 2, 2, 4]},
                                    dpi=200)
        if n_rows == 1:
            axs = np.expand_dims(axs, 0)

        fig.suptitle("At the highlighted image location (pixel marked in RED), the model predicted Class X.\n" + \
                     "The explanation shows a prototype, that (according to the model) looks\n" + \
                     "similar to the image patch around the highlighed pixel.\n\n" + \
                     "Based on the explanation, do you think the model's prediction at the highlighted location is correct?")
        plt.subplots_adjust(top=0.775)

        for r, row_axs in enumerate(axs):
            for c, ax in enumerate(row_axs):
                ax.set_xticks([])
                ax.set_yticks([])

        for r, row_axs in enumerate(axs):
            for c, ax in enumerate(row_axs):
                if c == 0:
                    if r == 0:
                        ax.set_title(f"Image")
                        ax.imshow(img_with_region)
                    else:
                        ax.axis('off')
                
                if c == 1:
                    if r == 0:
                        ax.set_title(f"Relevant Pixel")
                        ax.imshow(img_region_with_pixel)
                        for spine in ax.spines.values():
                            spine.set_color('yellow')
                            spine.set_linewidth(2)
                    else:
                        ax.axis('off')
                
                if c == 2:     
                    if r == 0:
                        ax.set_title("Prototype")
                    pt_image_path = prototype_dir + f"prototype-img_{nearest_protos_pixel[r]}_gt.png"
                    pt_image = Image.open(pt_image_path)
                    if max(pt_image._size) > 65 * 513/224:
                        crop_size_hw = list(pt_image._size)[::-1]  # crop_size is (h, w) but PIL image size is (w, h)
                        if crop_size_hw[0] > 65 * 513/224:
                            crop_size_hw[0] = round(65 * 513/224)
                        if crop_size_hw[1] > 65 * 513/224:
                            crop_size_hw[1] = round(65 * 513/224)
                        pt_image = transforms.CenterCrop(size=crop_size_hw)(pt_image)

                    pt_image = np.asarray(pt_image)
                    
                    ax.imshow(pt_image)
                    #ax.set_xlabel(f"P{nearest_protos_pixel[r]}, Sim {nearest_protos_distances[r]:.2f}")
                    ax.set_xlabel(f"Sim. {nearest_protos_distances[r]:.2f}")

                    patch = patches.ConnectionPatch((1.1, 0.5-0.05*r), (-0.1, 0.5), coordsA=axs[0, 1].transAxes, coordsB=ax.transAxes, arrowstyle="->")
                    fig.patches.append(patch)
                    if r == 0:
                        plt.annotate("looks like", (0.5, 0.5), xycoords=patch, ha="center", va="bottom", fontsize='small')
                
                ax.set_xticks([])
                ax.set_yticks([])
        plt.tight_layout()

        plt.savefig(os.path.join(hive_vis_dir, f"{os.path.splitext(img_file)[0]}.png"))
        plt.close()


if __name__ == '__main__':
    argh.dispatch_command(run_evaluation)
