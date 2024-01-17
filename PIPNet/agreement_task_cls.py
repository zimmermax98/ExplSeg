import random
import numpy as np
import os, shutil
from PIL import Image, ImageDraw as D
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pylab as pylab
import torch
from torch import nn
from tqdm import tqdm

from pipnet.pipnet import PIPNet
from pipnet.test import expl_pipnet_cls
from util.args import get_args, get_device
from util.data import get_dataloaders
from util.func import get_patch_size
from util.vis_pipnet import visualize_topk, get_img_coordinates, topk_images



configs = [
    {"n_protos": 1, "mode": "default"},
    {"n_protos": 1, "mode": "no_proto_image"},
    {"n_protos": 1, "mode": "only_class_image"},
    {"n_protos": 3, "mode": "default"}
]

def main(args):
    n_protos = configs[args.config]["n_protos"]
    mode = configs[args.config]["mode"]

    args.run_name = args.state_dict_dir_net.split("/")[-3]
    args.log_dir = os.path.join(args.log_base_dir, args.run_name)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device, device_ids = get_device(args)

    trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device)
    if len(classes)<=20:
        if args.validation_size == 0.:
            print("Classes: ", testloader.dataset.class_to_idx)
        else:
            print("Classes: ", str(classes))

    net = PIPNet(args, num_classes=len(classes))
    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids = device_ids)

    # Load model
    with torch.no_grad():
        checkpoint = torch.load(args.state_dict_dir_net,map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'],strict=True) 
        print("Pretrained network loaded")
        net.module._classification.normalization_multiplier.requires_grad = False

    with torch.no_grad():
        xs1, _, _ = next(iter(trainloader))
        xs1 = xs1.to(device)
        _, pfs_locpooled, _, _, _ = net(xs1)
        wshape = pfs_locpooled.shape[-1]
        args.wshape = wshape #needed for calculating image patch size
        print("Output shape: ", pfs_locpooled.shape)

        del xs1, pfs_locpooled

    n_rows = 3  # always use n_rows=3 to keep the same plot scalings
    if mode == "only_class_image":
        n_cols = 2
    else:
        n_cols = 5 if mode == "no_proto_image" else 6  # add column for "looks like" arrow
    
    if mode == "only_class_image":
        topk_dict = topk_images(args, net, projectloader, device, use_precomp_topk=True)
    else:
        topk_dict = visualize_topk(net, projectloader, device, '', args, 
                                   save_vis=False, use_precomp_topk=True)

    
    img_idxs = np.arange(len(test_projectloader.dataset))
    examples = [{"img_idx": img_idx} for img_idx in img_idxs[:args.num_examples]]

    examples = expl_pipnet_cls(net, test_projectloader, device, examples)

    if args.random_examples:
        random.shuffle(examples)

    hive_vis_dir = os.path.join(args.log_dir, "agreement_task_cls")
    if os.path.exists(hive_vis_dir):
        shutil.rmtree(hive_vis_dir)
    os.makedirs(hive_vis_dir)

    resize_fn = torchvision.transforms.Resize(size=(args.image_size, args.image_size))
    for example in tqdm(examples, ncols=0):
        img_idx = example["img_idx"]
        img_path = test_projectloader.dataset.imgs[img_idx][0]
        img_file_name = os.path.splitext(os.path.basename(img_path))[0]
        
        img = np.asarray(resize_fn(Image.open(img_path)))

        params = {'figure.autolayout': 'True',
                  'figure.titlesize': 'small',
                  'axes.labelsize': 'small',
                  'axes.titlesize': 'small'}
        pylab.rcParams.update(params)

        fig, axs = plt.subplots(n_rows, n_cols,
                                figsize=(1.6 * 6, (1.6 * n_rows)),
                                dpi=200)
        if n_rows == 1:
            axs = np.expand_dims(axs, 0)

        if mode == "only_class_image":
            fig.suptitle("For this image the model predicted Class X.\n" + \
                        "The explanation shows an image that (according to the model) contains\n" + \
                        "the same class as the one predicted.\n\n" + \
                        "Based on the explanation, do you think the model's prediction is correct?")
        else:
            fig.suptitle("For this image the model predicted Class X.\n" + \
                        "The explanation shows a prototype, that (according to the model) looks\n" + \
                        "similar to the image patch marked in yellow.\n\n" + \
                        "Based on the explanation, do you think the model's prediction is correct?")
        plt.subplots_adjust(top=0.775)

        for r, row_axs in enumerate(axs):
            for c, ax in enumerate(row_axs):
                ax.set_xticks([])
                ax.set_yticks([])

        for r, row_axs in enumerate(axs):
            if r >= min(len(example["topk_pfs"]), n_protos):
                for c, ax in enumerate(row_axs):
                    ax.axis("off")
            else:
                for c, ax in enumerate(row_axs):
                    if len(example["topk_pfs"])-1 >= r or (mode == "only_class_image" and r < 1):
                        if c == 0:
                            if r == 0:
                                ax.set_title("Image")
                                ax.imshow(img)
                            else:
                                ax.axis('off')
                        if c == 1:
                            if mode == "only_class_image":
                                if r == 0:
                                    ax.set_title("Best Class Image")
                                topk_image_index = np.random.randint(10) if args.random_prototype else 0
                                topk_class_img_index = topk_dict[example["pred_class"]][topk_image_index] 
                                class_img_path = projectloader.dataset.imgs[topk_class_img_index][0]
                                class_img = np.asarray(resize_fn(Image.open(class_img_path)))
                                ax.imshow(class_img)       
                                
                                patch = patches.ConnectionPatch((1.1, 0.5), (-0.1, 0.5), coordsA=axs[r, 0].transAxes, coordsB=ax.transAxes, arrowstyle="->")
                                fig.patches.append(patch)
                                plt.annotate("looks like", (0.5, 0.5), xycoords=patch, ha="center", va="bottom", fontsize='small')
                            else:
                                if r == 0:
                                    ax.set_title("Image with Region")
                                
                                h_idx, w_idx = example["topk_pfs_pooled_idxs"][r]
                                patchsize, skip = get_patch_size(args)
                                h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, args.wshape, patchsize, skip, h_idx, w_idx)
                                
                                img_with_region = Image.fromarray(img.copy())
                                draw = D.Draw(img_with_region)
                                draw.rectangle([
                                    (w_idx*skip,h_idx*skip), 
                                    (min(args.image_size, w_idx*skip+patchsize), 
                                     min(args.image_size, h_idx*skip+patchsize))], 
                                    outline='yellow', width=2)
                                img_with_region = np.asarray(img_with_region)
                                ax.imshow(img_with_region)

                        elif c == 2:     
                            if r == 0:
                                ax.set_title("Relevant Region")
                            
                            img_patch = img[h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                            
                            ax.imshow(img_patch)
                            for spine in ax.spines.values():
                                spine.set_color('yellow')
                                spine.set_linewidth(2)
                    
                        elif c == 3:
                            ax.axis("off")

                        elif c == 4:
                            if r == 0:
                                ax.set_title("Prototype")
                            
                            topk_proto_index = np.random.randint(10) if args.random_prototype else 0
                            topk_proto = topk_dict[example["topk_pfs"][r]][topk_proto_index] 
                            proto_img_img_idx, proto_img_h_idx, proto_img_w_idx, proto_img_sim_value = \
                                topk_proto["img_idx"], topk_proto["h_idx"], topk_proto["w_idx"], topk_proto["sim_value"]
                            proto_img_path = projectloader.dataset.imgs[proto_img_img_idx][0]
                            proto_img = np.asarray(resize_fn(Image.open(proto_img_path)))
                            patchsize, skip = get_patch_size(args)
                            h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, args.wshape, patchsize, skip, proto_img_h_idx, proto_img_w_idx)
                            proto_img_patch = proto_img[h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                            
                            ax.imshow(proto_img_patch)
                            ax.set_xlabel(f"Sim. {example['topk_sim_scores'][r]:.1f}")
                            for spine in ax.spines.values():
                                spine.set_color('yellow')
                                spine.set_linewidth(2)

                            patch = patches.ConnectionPatch((1.1, 0.5), (-0.1, 0.5), coordsA=axs[r, 2].transAxes, coordsB=ax.transAxes, arrowstyle="->")
                            fig.patches.append(patch)
                            plt.annotate("looks like", (0.5, 0.5), xycoords=patch, ha="center", va="bottom", fontsize='small')
                        
                        elif c == 5:
                            if r == 0:
                                ax.set_title("Prototype Image")
                            proto_img_with_region = Image.fromarray(proto_img)
                            draw = D.Draw(proto_img_with_region)
                            draw.rectangle([
                                (proto_img_w_idx*skip,proto_img_h_idx*skip), 
                                (min(args.image_size, proto_img_w_idx*skip+patchsize), 
                                 min(args.image_size, proto_img_h_idx*skip+patchsize))], 
                                outline='yellow', width=2)
                            proto_img_with_region = np.asarray(proto_img_with_region)
                            ax.imshow(proto_img_with_region)

        is_correct = (example["pred_class"] == example["true_class"])
        save_dir = os.path.join(hive_vis_dir, f"config_{mode}", 'correct' if is_correct else 'incorrect')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{img_file_name}.png"), bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    args = get_args(args_config="agreement_task")
    main(args)
