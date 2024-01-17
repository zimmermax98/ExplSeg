import copy

import click
import numpy as np
import torch
from crp.helper import get_layer_names
from crp.image import imgify
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.pylab as pylab
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from PIL import ImageDraw as D
from scipy.ndimage import distance_transform_bf
from tqdm import tqdm

import os, shutil
import sys
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "ExplSeg", "L-CRP"))

from datasets import get_dataset
from datasets.voc2012 import VOCSegmentation
from models import get_model
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes, make_grid
import zennit.image as zimage

from utils.crp import ChannelConcept
from utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from utils.render import vis_opaque_img_border

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "ExplSeg", "DeepLabV3", "datasets"))
from cityscapes import Cityscapes

def cityscapesID2TrainID(id):
    for c in Cityscapes.classes:
        if c.id == id:
            return c.train_id

EXAMPLES = [{
        "model_name": "deeplabv3plus",
        "dataset_name": "voc2012",
        "sample_id": 401,
        "class_id": 13,
        "layer": "backbone.layer4.0.conv3"},
    """{
        "model_name": "unet",
        "dataset_name": "cityscapes",
        "sample_id": 22,
        "class_id": 12,
        "layer": "encoder.features.15"},
    {
        "model_name": "yolov6",
        "dataset_name": "coco2017",
        "sample_id": 140,
        "class_id": 0,
        "layer": "backbone.ERBlock_5.0.rbr_dense.conv"},
    {
        "model_name": "yolov5",
        "dataset_name": "coco2017",
        "sample_id": 140,
        "class_id": 0,
        "layer": "model.8.cv3.conv"},
    {
        "model_name": "ssd",
        "dataset_name": "coco",
        "sample_id": 195,
        "class_id": 1,
        "layer": "model.backbone.vgg.28"},"""
]

EXAMPLE = 0

@click.command()
@click.option("--model_name", default=EXAMPLES[EXAMPLE]["model_name"])
@click.option("--dataset_name", default=EXAMPLES[EXAMPLE]["dataset_name"])
@click.option("--sample_id", default=EXAMPLES[EXAMPLE]["sample_id"], type=int)
@click.option("--class_id", default=EXAMPLES[EXAMPLE]["class_id"])
@click.option("--layer", default=EXAMPLES[EXAMPLE]["layer"], type=str)
@click.option("--prediction_num", default=0)
@click.option("--mode", default="relevance")
@click.option("--n_concepts", default=3)
@click.option("--n_refimgs", default=12)
def main(model_name, dataset_name, sample_id, class_id, layer, prediction_num, mode, n_concepts, n_refimgs):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset = test_dataset(preprocessing=True)
    dataset_len = len(dataset)

    val_images = VOCSegmentation(root="/fastdata/MT_ExplSeg/datasets/VOC", year="2012", image_set='val', download=False, transform=None).images

    model = get_model(model_name=model_name, classes=n_classes)

    model = model.to(device)
    model.eval()

    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])

    N_CONCEPTS_VIS = 3
    SHOW_CONCEPT_ORIG_IMG = False
    SAMPLE_CORRECT = True
    
    save_path = f"./output/crp/{model_name}_{dataset_name}/explanations_{'correct' if SAMPLE_CORRECT else 'incorrect'}/"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    for sample_id in tqdm(range(dataset_len), ncols=0):
        if dataset.images[sample_id] not in val_images:
            continue
        img, ann = dataset[sample_id]
        ann = ann.numpy()
        img = img[None, ...].to(device)
        #condition = [{"y": class_id}]

        pred = model(img).squeeze(0).argmax(0).cpu().numpy()

        if SAMPLE_CORRECT:
            # correct = 1, incorrect = 0
            correct = (pred == ann)
            
            # set "ignore" pixels to 0, so that distant correct pixels are also far away from "ignore" regions
            correct[ann == 255] = 0

            # ignore background pixels
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
            pred_classes, pred_class_count = np.unique(pred, return_counts=True)

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
            
            incorrect = (pred == incorr_pred_classes[argmax]) & (ann != 255)
                
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

        pixel_xy = (max_dist_h_idx, max_dist_w_idx)

        condition = [{"pixel": pixel_xy}]


        n_rows = N_CONCEPTS_VIS
        n_cols = 4 if SHOW_CONCEPT_ORIG_IMG else 3

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
                     "The explanation shows a concept example, that (according to the model) looks\n" + \
                     "similar to the image patch around the highlighed pixel.\n\n" + \
                     "Based on the explanation, do you think the model's prediction at the highlighted location is correct?")
        plt.subplots_adjust(top=0.775)

        if "deeplab" in model_name or "unet" in model_name:
            attr = attribution(copy.deepcopy(img).requires_grad_(), condition, composite, record_layer=[layer],
                            init_rel=1)

            resize_fn = torchvision.transforms.Resize(size=(224, 224))
            region_size = 32
            #img_resize = resize_fn(img).squeeze(0).permute(1, 2, 0).cpu().numpy()
            pixel_y, pixel_x = pixel_xy[0], pixel_xy[1]
            img_path = dataset.images[sample_id]
            img_resize = np.array(resize_fn(Image.open(img_path)))
            pixel_y_downscale, pixel_x_downscale = round(pixel_y * 224/513), round(pixel_x * 224/513)
            region_h_min = max(0, pixel_y_downscale - region_size)
            region_h_max = min(224-1, pixel_y_downscale + region_size+1)
            region_w_min = max(0, pixel_x_downscale - region_size)
            region_w_max = min(224-1, pixel_x_downscale + region_size+1)
            img_region = img_resize[region_h_min:region_h_max, region_w_min:region_w_max]
            img_region_with_pixel = Image.fromarray(img_region.astype(np.uint8))
            draw = D.Draw(img_region_with_pixel)
            draw.rectangle([(region_size-2, region_size-2), (region_size+2, region_size+2)], outline="red", fill=None, width=1)
            img_region_with_pixel = np.asarray(img_region_with_pixel)

            img_with_region = Image.fromarray(img_resize.astype(np.uint8))
            draw = D.Draw(img_with_region)
            draw.rectangle([(pixel_x_downscale-region_size-2, pixel_y_downscale-region_size-2), (pixel_x_downscale+region_size+2, pixel_y_downscale+region_size+2)], outline="yellow", fill=None, width=3)
            img_with_region = np.asarray(img_with_region)
            
        else:
            raise NameError

        layer_map = get_layer_names(model, [torch.nn.Conv2d])
        fv = VISUALIZATIONS[model_name](attribution, dataset, layer_map, preprocess_fn=lambda x: x,
                                        path=f"output/crp/{model_name}_{dataset_name}",
                                        max_target="max", device=device)

        if mode == "relevance":
            channel_rels = torch.nn.functional.interpolate(attr.relevances[layer], img_resize.shape[:2], mode="bilinear")[..., pixel_y_downscale, pixel_x_downscale]

        topk = torch.topk(channel_rels[0], n_concepts)
        topk_ind = topk.indices.detach().cpu().numpy()
        topk_rel = topk.values.detach().cpu().numpy()

        ref_imgs, ref_imgs_without_heatmap = fv.get_max_reference(topk_ind, layer, mode, (0, n_refimgs), composite=composite, rf=True,
                                        plot_fn=vis_opaque_img_border, batch_size=4)

        resize = torchvision.transforms.Resize((150, 150))
        resize_orig = torchvision.transforms.Resize((513, 513))

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
                        ax.set_title("Concept")
                    grid = False
                    if grid:
                        grid = make_grid(
                            #[resize(torch.from_numpy(np.array(i)).permute((2, 0, 1))) for i in ref_imgs[topk_ind[r]]],
                            [resize(i) for i in ref_imgs_without_heatmap[topk_ind[r]]],
                            nrow=int(n_refimgs / 2),
                            padding=0)
                        grid = np.array(zimage.imgify(grid.detach().cpu()))
                        ax.imshow(grid)
                    else:
                        concept_index = np.random.randint(12)
                        ax.imshow(resize_orig(ref_imgs_without_heatmap[topk_ind[r]][concept_index]).permute(1, 2, 0).cpu().numpy())
                    
                    patch = patches.ConnectionPatch((1.1, 0.5-0.05*r), (-0.1, 0.5), coordsA=axs[0, 1].transAxes, coordsB=ax.transAxes, arrowstyle="->")
                    fig.patches.append(patch)
                    if r == 0:
                        plt.annotate("looks like", (0.5, 0.5), xycoords=patch, ha="center", va="bottom", fontsize='small')


        plt.tight_layout()
        print()
        #plt.show()
        img_basename = dataset.images[sample_id].rsplit("/", 1)[1].rsplit(".", 1)[0]
        plt.savefig(os.path.join(save_path, f"{img_basename}.png"))
        plt.close()

if __name__ == "__main__":
    main()
