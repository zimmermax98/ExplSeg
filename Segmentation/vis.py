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

def get_argparser():
    parser = argparse.ArgumentParser()

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

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

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

def save_image(img, img_path, patches, with_legend):
    if with_legend:
        plt.figure(figsize=(img.shape[1] / 50, img.shape[0] / 50))
        plt.imshow(img)
        plt.tight_layout()
        plt.axis('off')
        plt.legend(handles=patches)
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
    else:
        Image.fromarray(img).save(img_path)


def main():
    opts = get_argparser().parse_args()

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
    
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)

    with torch.no_grad():
        model.eval()
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

            original_img = val_data.unnorm_transform(img, label)[0][0].permute(1, 2, 0).cpu().numpy() * 255
            if opts.vis_type == "pred" or opts.vis_type == "all":
                colorized_preds = decode_fn(pred).astype('uint8')
                colorized_img = (0.5 * colorized_preds + 0.5 * original_img).astype('uint8')
                patches = [mpatches.Patch(color=tuple(decode_fn(pred_class) / 255), label=val_loader.dataset.class_idx_to_name(pred_class)) 
                        for pred_class in np.unique(pred)]
                
                save_path = os.path.join(opts.save_val_results_to, f"{img_name}_pred.png")
                save_image(colorized_img, save_path, patches, opts.with_legend)

            if opts.vis_type == "label" or opts.vis_type == "all":
                colorized_labels = decode_fn(label).astype('uint8')
                colorized_img = (0.5 * colorized_labels + 0.5 * original_img).astype('uint8')
                patches = [mpatches.Patch(color=tuple(decode_fn(pred_class) / 255), label=val_loader.dataset.class_idx_to_name(pred_class)) 
                        for pred_class in np.unique(label)]
                
                save_path = os.path.join(opts.save_val_results_to, f"{img_name}_label.png")
                save_image(colorized_img, save_path, patches, opts.with_legend)

            if opts.vis_type == "correct" or opts.vis_type == "all":
                correct = (pred == label).astype(np.uint8)
                correct[label == 255] = 2  # invalid pixels
                
                correct_cmap = np.array([[179, 0, 0, 255],  # incorrect
                                         [0, 179, 0, 255],  # correct
                                         [0, 0, 0, 0]])     # ignore
                colorized_correct = correct_cmap[correct]
                original_img_rgba = np.dstack((original_img, np.full_like(original_img[..., 0], 255)))
                colorized_img = (0.5 * colorized_correct + 0.5 * original_img_rgba).astype('uint8')
                
                patches = [mpatches.Patch(color=tuple(correct_cmap[k] / 255), label=v) 
                        for k, v in enumerate(["incorrect", "correct", "ignore"])]
                
                save_path = os.path.join(opts.save_val_results_to, f"{img_name}_correct.png")
                save_image(colorized_img, save_path, patches, opts.with_legend)

    score = metrics.get_results()
    print(metrics.to_str(score))

if __name__ == '__main__':
    main()
