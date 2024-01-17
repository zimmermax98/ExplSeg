from tqdm import tqdm
import network
import utils
import os
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
from torch import nn

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from glob import glob

from main import get_dataset

import cv2
from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution
from captum.attr import visualization

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--attr_method", type=str, default='segintgrad',
                    choices=['random', 'segintgrad', 'seggradcam'], help='method name')
    parser.add_argument("--maskMode", type=str, required=False,
                        choices=["pixelMask", "rectMask", "classMask"], default="classMask")
    parser.add_argument("--target", type=int, required=False, default=-2,
                        help="Index of class that the attribution should be calculated for. \
                        '-1' for all classes, '-2' for predicted class")

    # Datset Options
    parser.add_argument("--vis_type", type=str, required=False,
                        choices=["pred", "label", "correct", "all"], default="all")
    parser.add_argument("--with_legend", action='store_true', default=False)
    parser.add_argument("--input_type", type=str, required=False,
                        choices=["data_loader", "path"], default="data_loader")
    parser.add_argument("--input", type=str, required=False,
                        help="path to a single image or image directory")
    parser.add_argument("--data_root", type=str, default='/fastdata/MT_ExplSeg/datasets',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    
    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')
    return parser


class ClassFilterLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_batch, target):
        logits_batch_filtered = []
        for i_batch in range(logits_batch.shape[0]):
            logits = logits_batch[i_batch]
            logits = logits.flatten(start_dim=1, end_dim=-1)
            class_indices = logits.argmax(dim=0, keepdim=True)
            class_indices = torch.cat([class_indices]*logits.shape[0], dim=0)
            logits_filtered = logits[class_indices == target].view(logits.shape[0], -1)
            logits_batch_filtered.append(logits_filtered.sum(1).unsqueeze(0))
        return torch.cat(logits_batch_filtered, dim=0)
    
class PixelFilterLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_batch, target):
        y, x = target
        return logits_batch[:, :, y, x]
        
class RectFilterLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_batch, target):
        ymin, ymax, xmin, xmax = target
        return torch.sum(logits_batch[:, :, ymin:ymax, xmin:xmax], dim=[2, 3])
    
class MaskedSegmentationModel(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.segmentation_model = model

    def forward(self, input, mask):
        logits = self.segmentation_model(input)
        # classMask is scalar (classID), pixelMask is [x, y]
        if hasattr(mask, "__len__"):
            if len(mask) == 2:
                filterLayer = PixelFilterLayer() 
            elif len(mask) == 4:
                filterLayer = RectFilterLayer()
        else:
            filterLayer = ClassFilterLayer()
        
        logits_filtered = filterLayer(logits, mask)
        return logits_filtered
    

def cityscapesTrainID2Name(trainID):
    for c in Cityscapes.classes:
        if c.train_id == trainID:
            return c.name

def cityscapesName2TrainID(name):
    for c in Cityscapes.classes:
        if c.name == name:
            return c.train_id
        
def process_attributions(attr, attr_method, mode="abs"):
    attr = np.sum(attr, axis=-1)
    if mode == "all":
        pass
    elif mode == "pos":
        attr = np.maximum(0, attr)
    elif mode == "abs":
        attr = np.abs(attr)

    min = np.min(attr)
    if attr_method == "segintgrad":
        max = np.max(attr)
    elif attr_method == "seggradcam":
        max = np.max(np.percentile(attr, 99.5, axis=(1, 2)))
    attr_norm = (attr - min) / (max - min)  # normalize between 0 and 1
    attr_norm = np.clip(attr_norm, 0, 1)
    if mode == "all":
        attr_norm = (attr_norm * 2) - 1
    return attr_norm 

def visualize_attribution(attribution_map, original_img=None):
    heatmap = cv2.applyColorMap(np.uint8(255 * attribution_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if original_img is not None:
        heatmap = 0.5 * heatmap + 0.5 * original_img
    return np.uint8(255 * heatmap)

def main():
    opts = get_argparser().parse_args()
    opts.data_root = "/Volumes/ExtM2/MT_ExplSeg/datasets"
    torch.manual_seed(1)
    np.random.seed(1)

    pixelMasks = [  # y, x
        [100, 100]
    ]

    rectMasks = [  # y_min, y_max, x_min, x_max
        [100, 200, 100, 200]
    ]

    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        opts.data_root = os.path.join(opts.data_root, "VOC")
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        opts.data_root = os.path.join(opts.data_root, "cityscapes")
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    if opts.input_type == "data_loader":
        val_data = get_dataset(opts)[1]
        val_loader = data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1)
    elif opts.input_type == "path":
        image_files = []
        if os.path.isdir(opts.input):
            for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
                files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
                if len(files)>0:
                    image_files.extend(files)
        elif os.path.isfile(opts.input):
            image_files.append(opts.input)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    print("Model restored from %s" % opts.ckpt)
    del checkpoint  # free memory

    model_mod = MaskedSegmentationModel(model).to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.input_type == "path":
        if opts.crop_val:
            transform = T.Compose([
                    T.Resize(opts.crop_size),
                    T.CenterCrop(opts.crop_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
        else:
            transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
    else:
        # we can only calculate metrics when we have a dataset with ground truth labels
        metrics = StreamSegMetrics(opts.num_classes)
        metrics.reset()

    if opts.dataset == "cityscapes":
        train_classes = np.delete(np.unique([cls.train_id for cls in val_data.classes]), -1)  # subtract void class (255)
        n_classes = len(train_classes)
    elif opts.dataset == "voc":
        n_classes = len(val_data.class_names)

    if opts.attr_method == 'segintgrad':
        ig = IntegratedGradients(model_mod)
        baseline = torch.zeros_like(next(iter(val_loader))[0]).to(device)
    elif opts.attr_method == 'seggradcam':
        #grad_cam_layer = model_mod.segmentation_model.module.backbone.high_level_features[-1]
        grad_cam_layer = model_mod.segmentation_model.module.backbone.layer4[0].conv1
        gradcam = LayerGradCam(model_mod, layer=grad_cam_layer)

    classMasksIDs = np.arange(n_classes)

    if opts.maskMode == "rectMask":
        masks = rectMasks
    elif opts.maskMode == "pixelMask":
        masks = pixelMasks
    elif opts.maskMode == "classMask":
        masks = classMasksIDs

    if opts.target == -1:
        targetIds = np.arange(n_classes)
    elif opts.target == -2:
        pass
    else:
        targetIds == [opts.target]

    with torch.no_grad():
        model.eval()
        model_mod.eval()
        if opts.input_type == "path":
            data_iterable = image_files
        else:
            data_iterable = val_loader
        for i, (img_path_or_img_label) in tqdm(enumerate(data_iterable)):
            if opts.input_type == "path":
                img_path = img_path_or_img_label
                img = Image.open(img_path).convert('RGB')
                img = transform(img).unsqueeze(0) # To tensor of NCHW
                img = img.to(device)
            else:
                img, label = img_path_or_img_label
                img, label = img.to(device), label.to(device)
                img_path = val_data.images[i]
            
            img_name, ext = os.path.splitext(os.path.basename(img_path))
            if opts.dataset == "cityscapes":
                img_name = img_name.replace("_leftImg8bit", "")

            pred = model(img).argmax(1)
            pred, label = pred.cpu().numpy(), label.cpu().numpy()
            if opts.input_type == "data_loader":
                metrics.update(label, pred)
                pred, label = pred[0], label[0]

            for mask in masks:
                if opts.maskMode == "rectMask":
                    outputID = pred[mask[0]:mask[1], mask[2]:mask[3]].argmax()
                elif opts.maskMode == "pixelMask":
                    outputID = pred[mask[0], mask[1]]
                elif opts.maskMode == "classMask":
                    outputID = mask

                if opts.dataset == "cityscapes":
                    outputClassName = cityscapesTrainID2Name(outputID)
                elif opts.dataset == "voc":
                    outputClassName = val_data.class_names[outputID]

                if opts.target == -2:
                    targetIds = [outputID]

                attributions = {}
                for targetId in targetIds:
                    if opts.dataset == "cityscapes":
                        targetClassName = cityscapesTrainID2Name(targetId)
                    elif opts.dataset == "voc":
                        targetClassName = val_data.class_names[targetId]
                    
                    # get attribution for sample
                    if opts.attr_method == 'random':
                        attribution = torch.rand(img.shape).to(device) - 0.5  # center around 0
                    elif opts.attr_method == 'segintgrad':
                        attribution = ig.attribute(img,
                                                additional_forward_args=mask,
                                                target=torch.tensor(targetId),
                                                baselines=baseline,
                                                method='gausslegendre',
                                                n_steps=128,
                                                internal_batch_size=opts.val_batch_size,
                                                return_convergence_delta=False)
                        attribution = attribution.to(device).float().detach()
                    elif opts.attr_method == 'seggradcam':
                        attribution_layer = gradcam.attribute(img,
                                                        additional_forward_args=mask,
                                                        target=torch.tensor(targetId),
                                                        attribute_to_layer_input=False,
                                                        relu_attributions=True,
                                                        attr_dim_summation=1)
                        attribution_layer = attribution_layer.to(device).float().detach()
                        attribution_layer = attribution_layer - attribution_layer.min()
                        attribution_layer = attribution_layer / (1e7 + attribution_layer.max())

                        attribution = LayerAttribution.interpolate(attribution_layer, 
                                                                [img.shape[-2], img.shape[-1]], 
                                                                interpolate_mode="bilinear")
                
                    attribution = attribution.squeeze(0).cpu().permute(1, 2, 0).numpy()
                    if np.all(attribution == 0):
                        break
                    attributions[targetClassName] = attribution
                
                else:  # only runs if inner loop didn't break
                    attribution_values = np.array(list(attributions.values()))
                    
                    attribution_values = process_attributions(attribution_values, opts.attr_method, mode="abs")
                    attributions = dict(zip(attributions.keys(), attribution_values))
                    for (targetClassName, attribution) in attributions.items():
                        if opts.input_type == "data_loader":
                            val_img = val_data.unnorm_transform(img, label)[0].squeeze(0).cpu().permute(1, 2, 0).numpy()
                        attribution_vis = visualize_attribution(attribution, original_img=val_img)
                        pil_image = Image.fromarray(attribution_vis)
                        save_file_dir = os.path.join("attribution_results", opts.dataset, opts.maskMode, img_name)
                        save_file_name = f"{img_name}_{opts.attr_method}_of_{outputClassName}_for_{targetClassName}.png"
                        if not os.path.exists(save_file_dir):
                            os.makedirs(save_file_dir)
                        pil_image.save(os.path.join(save_file_dir, save_file_name))

if __name__ == '__main__':
    main()
