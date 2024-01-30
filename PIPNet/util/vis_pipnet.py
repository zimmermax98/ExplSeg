from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
import torch.utils.data
import os, shutil
from PIL import Image, ImageDraw as D
from torchvision import transforms
import torchvision
from util.func import get_patch_size
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

@torch.no_grad()                  
def visualize_topk_cls(net, projectloader, device, foldername, args: argparse.Namespace, k=10, save_vis=True, use_precomp_topk=False):
    if args.dataset == "VOC":
        imgs = projectloader.dataset.images
        class2name = projectloader.dataset.class_idx_to_name
    else:
        imgs = projectloader.dataset.imgs
        if isinstance(imgs[0], tuple):
            imgs = [path for path, label in imgs]
        class2name = {i: v for i, v in enumerate(projectloader.dataset.classes)}.get
    
    pretrain = ('pretrain' in foldername)

    num_prototypes = net.module._num_prototypes
    
    patchsize, skip = get_patch_size(args)

    net.eval()
    classification_weights = net.module._classification.weight.squeeze()
    
    topk_json_file_path = os.path.join(args.log_dir, "topk_prototypes.json")
    
    if os.path.exists(topk_json_file_path) and use_precomp_topk:
        print("Reading in topk prototypes from file")
        with open(topk_json_file_path) as f:
            topk_dict = json.load(f)
        # convert keys that were read in as str back to int
        topk_dict = {int(key): value for key, value in topk_dict.items()}
        for proto_key, topk_value_dict in topk_dict.items():
            topk_dict[proto_key] = {int(key): value for key, value in topk_value_dict.items()}
    
    else:
        # Iterate through the training set
        pfs_pooled_list = []
        pooled_idxs_list = []
        abstained = 0
        for xs, ys in tqdm(projectloader, desc='Collecting topk prototypes', ncols=0):
            xs, ys = xs.to(device), ys.to(device)

            with torch.no_grad():
                # Use the model to classify this batch of input data
                pfs, pfs_locpooled, (pfs_pooled, pooled_idxs), _, out = net(xs, inference=True)
                pfs_pooled_list.append(pfs_pooled)
                pooled_idxs_list.append(pooled_idxs)
                outmax, _ = torch.max(out[0], dim=0) #shape ([1]) because batch size of projectloader is 1
                abstained += torch.sum(outmax == 0.).item()
                del pfs, pfs_locpooled, pfs_pooled, pooled_idxs, out

        print("Abstained patches: ", abstained)
        
        pooled = torch.cat(pfs_pooled_list).T
        pooled_idxs = torch.cat(pooled_idxs_list).T

        pooled_h_idxs, pooled_w_idxs = np.unravel_index(pooled_idxs.cpu().numpy(), (args.wshape, args.wshape))
        topk_values, topk_img_idxs = torch.topk(pooled, k, dim=1, largest=True, sorted=True)
        topk_values, topk_img_idxs = topk_values.cpu().numpy(), topk_img_idxs.cpu().numpy()
        row_idxs = np.arange(net.module._num_prototypes).reshape((-1, 1))
        topk_h_idxs, topk_w_idxs = pooled_h_idxs[row_idxs, topk_img_idxs], pooled_w_idxs[row_idxs, topk_img_idxs]

        topk_dict = {}
        for p in range(num_prototypes):
            p_dict = {}
            for i in range(k):
                img_idx = topk_img_idxs[p, i]
                h_idx = topk_h_idxs[p, i]
                w_idx = topk_w_idxs[p, i]
                sim_value = topk_values[p, i]  # similarity
                p_dict[i] = {
                    "img_idx": img_idx,
                    "h_idx": h_idx,
                    "w_idx": w_idx,
                    "sim_value": sim_value
                }
            topk_dict[p] = p_dict

        topk_json = json.dumps(topk_dict, indent=4, cls=NpEncoder)
        with open(topk_json_file_path, "w") as f:
            f.write(topk_json)

    if not save_vis:
        return topk_dict

    dir = os.path.join(args.log_dir, foldername)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    prototype_iter = tqdm(range(num_prototypes),
                          desc='Visualizing topk',
                          ncols=0)

    all_prototype_patches = []
    resize_fn = transforms.Resize(size=(args.image_size, args.image_size))
    class_weight_threshold = 1  # original 1e-10
    for p in prototype_iter:
        cls_weights_p = classification_weights[:, p]
        if ((cls_weights_p.max() < class_weight_threshold) and not pretrain):
            continue
        if topk_dict[p][0]["sim_value"] == 0:  # if even the best similarity score is 0
            continue
        
        prototype_patches = []
        for i in range(k):
            img_idx = topk_dict[p][i]["img_idx"]
            h_idx = topk_dict[p][i]["h_idx"]
            w_idx = topk_dict[p][i]["w_idx"]

            # extract image patch from the region
            img = torchvision.transforms.ToTensor()(resize_fn(Image.open(imgs[img_idx])))
            h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, args.wshape, patchsize, skip, h_idx, w_idx)
            img_patch = img[:, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
            prototype_patches.append(img_patch)
        
        # add text next to each topk-grid, to easily see which prototype it is
        txtpatch_pil = Image.new("RGB", (img_patch.shape[1],img_patch.shape[2]), (0, 0, 0))
        D.Draw(txtpatch_pil).text((1, img_patch.shape[2]//2), f"P{p}", anchor='lm', fill="white")
        txtpatch = torchvision.transforms.ToTensor()(txtpatch_pil)
        prototype_patches.insert(0, txtpatch)
        
        # save top-k image patches in grid
        grid = torchvision.utils.make_grid(prototype_patches, nrow=k+1, padding=1)
        image_name = f"grid_topk_{p}"
        if not pretrain:
            rel_classes = torch.where(cls_weights_p >= class_weight_threshold)[0]
            rel_classes_sorted = rel_classes[torch.argsort(-cls_weights_p[rel_classes])]
            for rel_class in rel_classes_sorted:
                cls_name = class2name(rel_class.item())
                cls_weight = cls_weights_p[rel_class]
                image_name += f"_{cls_name}{cls_weight:.1f}"
        torchvision.utils.save_image(grid,os.path.join(dir, f"{image_name}.png"))
        all_prototype_patches += prototype_patches

    if len(all_prototype_patches)>0:
        grid = torchvision.utils.make_grid(all_prototype_patches, nrow=k+1, padding=1)
        torchvision.utils.save_image(grid,os.path.join(dir, "grid_topk_all.png"))
    else:
        print("Pretrained prototypes not visualized. Try to pretrain longer.")
    
    return topk_dict


@torch.no_grad()                    
def visualize_topk_seg(net, projectloader, device, foldername, args: argparse.Namespace, k=10, save_vis=True, use_precomp_topk=False):
    imgs = projectloader.dataset.images
    class2name = projectloader.dataset.class_idx_to_name
    
    pretrain = ('pretrain' in foldername)

    num_prototypes = net.module._num_prototypes
    
    patchsize, skip = get_patch_size(args)
    
    net.eval()
    classification_weights = net.module._classification.weight.squeeze()
    
    topk_json_file_path = os.path.join(args.log_dir, "topk_prototypes.json")
    
    if os.path.exists(topk_json_file_path) and use_precomp_topk:
        print("Reading in topk prototypes from file")
        with open(topk_json_file_path) as f:
            topk_dict = json.load(f)
        # convert keys that were read in as str back to int
        topk_dict = {int(key): value for key, value in topk_dict.items()}
        for proto_key, topk_value_dict in topk_dict.items():
            topk_dict[proto_key] = {int(key): value for key, value in topk_value_dict.items()}
    
    else:    
        # Iterate through the training set
        pfs_pooled_list = []
        pooled_idxs_list = []
        abstained = 0
        for i, (xs, ys) in tqdm(enumerate(projectloader), total=len(projectloader), desc='Collecting topk prototypes', ncols=0):
            xs, ys = xs.to(device), ys.to(device)

            with torch.no_grad():
                # Use the model to classify this batch of input data
                pfs, pfs_locpooled, (pfs_pooled, pooled_idxs), _, out = net(xs, inference=True)
                pfs_pooled_list.append(pfs_pooled)
                pooled_idxs_list.append(pooled_idxs)
                outmax, ys_pred = torch.max(out[0], dim=0) #shape ([1]) because batch size of projectloader is 1
                abstained += torch.sum(outmax == 0.).item()
                del pfs, pfs_locpooled, pfs_pooled, pooled_idxs, out

        print("Abstained patches: ", abstained)  # TODO: Adapt for segmentation
        
        pooled = torch.cat(pfs_pooled_list).T
        pooled_idxs = torch.cat(pooled_idxs_list).T

        pooled_h_idxs, pooled_w_idxs = np.unravel_index(pooled_idxs.cpu().numpy(), (args.wshape, args.wshape))
        topk_values, topk_img_idxs = torch.topk(pooled, k, dim=1, largest=True, sorted=True)
        topk_values, topk_img_idxs = topk_values.cpu().numpy(), topk_img_idxs.cpu().numpy()
        row_idxs = np.arange(net.module._num_prototypes).reshape((-1, 1))
        topk_h_idxs, topk_w_idxs = pooled_h_idxs[row_idxs, topk_img_idxs], pooled_w_idxs[row_idxs, topk_img_idxs]
        
        topk_dict = {}
        for p in range(num_prototypes):
            p_dict = {}
            for i in range(k):
                img_idx = topk_img_idxs[p, i]
                h_idx = topk_h_idxs[p, i]
                w_idx = topk_w_idxs[p, i]
                sim_value = topk_values[p, i]  # similarity
                p_dict[i] = {
                    "img_idx": img_idx,
                    "h_idx": h_idx,
                    "w_idx": w_idx,
                    "sim_value": sim_value
                } 
            topk_dict[p] = p_dict

        topk_json = json.dumps(topk_dict, indent=4, cls=NpEncoder)
        with open(topk_json_file_path, "w") as f:
            f.write(topk_json)

    if not save_vis:
        return topk_dict

    dir = os.path.join(args.log_dir, foldername)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    prototype_iter = tqdm(range(num_prototypes),
                          total=num_prototypes,
                          desc='Visualizing topk',
                          ncols=0)

    all_prototype_patches = []
    resize_fn = transforms.Resize(size=(args.image_size, args.image_size))
    class_weight_threshold = 1  # original 1e-10
    for p in prototype_iter:
        cls_weights_p = classification_weights[:, p]
        if ((cls_weights_p.max() < class_weight_threshold) and not pretrain):
            continue
        if topk_dict[p][0]["sim_value"] == 0:  # if even the best similarity score is 0
            continue
        
        prototype_patches = []
        for i in range(k):
            img_idx = topk_dict[p][i]["img_idx"]
            h_idx = topk_dict[p][i]["h_idx"]
            w_idx = topk_dict[p][i]["w_idx"]

            # extract image patch from the region
            img = torchvision.transforms.ToTensor()(resize_fn(Image.open(imgs[img_idx]).convert('RGB')))
            h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, args.wshape, patchsize, skip, h_idx, w_idx)
            img_patch = img[:, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
            prototype_patches.append(img_patch)
        
        # add text next to each topk-grid, to easily see which prototype it is
        txtpatch_pil = Image.new("RGB", (img_patch.shape[1],img_patch.shape[2]), (0, 0, 0))
        D.Draw(txtpatch_pil).text((1, img_patch.shape[2]//2), f"P{p}", anchor='lm', fill="white")
        txtpatch = torchvision.transforms.ToTensor()(txtpatch_pil)
        prototype_patches.insert(0, txtpatch)

        # save top-k image patches in grid
        grid = torchvision.utils.make_grid(prototype_patches, nrow=k+1, padding=1)
        image_name = f"grid_topk_{p}"
        if not pretrain:
            rel_classes = torch.where(cls_weights_p >= class_weight_threshold)[0]
            rel_classes_sorted = rel_classes[torch.argsort(-cls_weights_p[rel_classes])]
            for rel_class in rel_classes_sorted:
                cls_name = class2name(rel_class)
                cls_weight = cls_weights_p[rel_class]
                image_name += f"_{cls_name}{cls_weight:.1f}"
        torchvision.utils.save_image(grid,os.path.join(dir, f"{image_name}.png"))
        all_prototype_patches += prototype_patches

    if len(all_prototype_patches)>0:
        grid = torchvision.utils.make_grid(all_prototype_patches, nrow=k+1, padding=1)
        torchvision.utils.save_image(grid,os.path.join(dir, "grid_topk_all.png"))
    else:
        print("Pretrained prototypes not visualized. Try to pretrain longer.")
    
    return topk_dict

def visualize_topk(net, projectloader, device, foldername, args: argparse.Namespace, k=10, save_vis=True, use_precomp_topk=False):
    if args.task == "classification":
        return visualize_topk_cls(net, projectloader, device, foldername, args, k, save_vis, use_precomp_topk)
    elif args.task == "segmentation":
        return visualize_topk_seg(net, projectloader, device, foldername, args, k, save_vis, use_precomp_topk)

def visualize_cls(net, projectloader, num_classes, device, foldername, args: argparse.Namespace):
    print("Visualizing prototypes...")
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    prototype_patches = dict()
    abstainedimgs = set()
    notabstainedimgs = set()
    
    for p in range(net.module._num_prototypes):
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p]=near_imgs_dir
        seen_max[p]=0.
        saved[p]=0
        saved_ys[p]=[]
        prototype_patches[p]=[]
    
    patchsize, skip = get_patch_size(args)

    if args.dataset == "VOC":
        imgs = projectloader.dataset.images
    else:
        imgs = projectloader.dataset.imgs
    
    # skip some images for visualisation to speed up the process
    if len(imgs)/num_classes <10:
        skip_img=10
    elif len(imgs)/num_classes < 50:
        skip_img=5
    else:
        skip_img = 2
    print("Every", skip_img, "is skipped in order to speed up the visualisation process")

    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    desc='Visualizing',
                    ncols=0)

    # Iterate through the data
    images_seen_before = 0
    for i, (xs, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        if i % skip_img == 0:
            images_seen_before+=xs.shape[0]
            continue
        
        xs, ys = xs.to(device), ys.to(device)
        # Use the model to classify this batch of input data
        with torch.no_grad():
            pfs, _, _, _, out = net(xs, inference=True) 

        max_per_prototype, max_idx_per_prototype = torch.max(pfs, dim=0)
        # In PyTorch, images are represented as [channels, height, width]
        max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
        max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1)
        for p in range(0, net.module._num_prototypes):
            c_weight = torch.max(classification_weights[:,p]) #ignore prototypes that are not relevant to any class
            if c_weight>0:
                h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]].item()
                w_idx = max_idx_per_prototype_w[p].item()
                idx_to_select = max_idx_per_prototype[p,h_idx, w_idx].item()
                found_max = max_per_prototype[p,h_idx, w_idx].item()

                imgname = imgs[images_seen_before+idx_to_select]
                if out.max() < 1e-8:
                    abstainedimgs.add(imgname)
                else:
                    notabstainedimgs.add(imgname)
                
                if found_max > seen_max[p]:
                    seen_max[p]=found_max
               
                if found_max > 0.5:
                    img_to_open = imgs[images_seen_before+idx_to_select]
                    if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): #dataset contains tuples of (img,label)
                        imglabel = img_to_open[1]
                        img_to_open = img_to_open[0]
                    else:
                        imglabel = projectloader.dataset.class_idx_to_name(ys)

                    image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_to_open).convert("RGB"))
                    img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, args.wshape, patchsize, skip, h_idx, w_idx)
                    img_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                    saved[p]+=1
                    prototype_patches[p].append((img_patch, found_max))
                    
                    save_path = os.path.join(dir, "prototype_%s")%str(p)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    draw = D.Draw(image)
                    draw.rectangle([(w_coor_min,h_coor_min), (w_coor_max, h_coor_max)], outline='yellow', width=2)
                    image.save(os.path.join(save_path, 'p%s_%s_%s_%s_rect.png'%(str(p),str(imglabel),str(round(found_max, 2)),str(img_to_open.split('/')[-1].split('.jpg')[0]))))
                    
        
        images_seen_before+=len(ys)

    print("num images abstained: ", len(abstainedimgs))
    print("num images not abstained: ", len(notabstainedimgs))
    for p in range(net.module._num_prototypes):
        if saved[p]>0:
            try:
                sorted_by_second = sorted(prototype_patches[p], key=lambda tup: tup[1], reverse=True)
                sorted_ps = [i[0] for i in sorted_by_second]
                grid = torchvision.utils.make_grid(sorted_ps, nrow=16, padding=1)
                torchvision.utils.save_image(grid,os.path.join(dir,"grid_%s.png"%(str(p))))
            except RuntimeError:
                pass

def visualize_seg(net, projectloader, num_classes, device, foldername, args: argparse.Namespace):
    print("Visualizing prototypes...")
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

    near_imgs_dirs = dict()
    saved = dict()
    saved_ys = dict()
    prototype_patches = dict()
    abstainedimgs = set()
    notabstainedimgs = set()
    
    for p in range(net.module._num_prototypes):
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p]=near_imgs_dir
        saved[p]=0
        saved_ys[p]=[]
        prototype_patches[p]=[]
    
    patchsize, skip = get_patch_size(args)

    imgs = projectloader.dataset.images
    
    # skip some images for visualisation to speed up the process
    if len(imgs)/num_classes <10:
        skip_img=10
    elif len(imgs)/num_classes < 50:
        skip_img=5
    else:
        skip_img = 2
    print("Every", skip_img, "is skipped in order to speed up the visualisation process")

    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight.squeeze()
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    desc='Visualizing',
                    ncols=0)

    # Iterate through the data
    assert projectloader.batch_size == 1
    for i, (xs, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        if i % skip_img == 0:
            continue
        
        xs = xs.to(device)
        ys = ys.to(device)
        # Use the model to classify this batch of input data
        with torch.no_grad():
            _, pfs_locpooled, _, out_no_upscale, out = net(xs, inference=True)
        pfs_locpooled, out_no_upscale, out, ys = pfs_locpooled[0], out_no_upscale[0], out[0], ys[0]  # batch_size = 1
        out_max_no_upscale, pred_no_upscale = out_no_upscale.max(dim=0)
        img_size_no_upscale = out_no_upscale.shape[1]
        
        imgname = imgs[i]
        if out.max() < 1e-8:
            abstainedimgs.add(imgname) # TODO: Adapt for segmentation
        else:
            notabstainedimgs.add(imgname)

        pred_classes = torch.unique(pred_no_upscale)
        out_max_no_upscale_flat = out_max_no_upscale.flatten()
        pred_no_upscale_flat = pred_no_upscale.flatten()

        for pred_class in pred_classes:
            # find the image patch where the class score for the predicted class is the heightest
            max_idx_class = (out_max_no_upscale_flat * (pred_no_upscale_flat == pred_class)).argmax().item()
            max_h_idx_class = max_idx_class // img_size_no_upscale
            max_w_idx_class = max_idx_class % img_size_no_upscale
            
            # only look at the prototypes that are relevant for that class
            class_pfs_idxs = torch.where(classification_weights[pred_class] > 0)[0]

            # find the prototype that had the highest similarity score at the selected image patch
            class_patch_pfs = pfs_locpooled[class_pfs_idxs, max_h_idx_class, max_w_idx_class]
            max_class_patch_pf, max_class_patch_pf_idx = torch.max(class_patch_pfs, dim=0)
            max_class_patch_pf = max_class_patch_pf.item()
            pf_index = class_pfs_idxs[max_class_patch_pf_idx].item()
            if max_class_patch_pf > 0.5:
                img_to_open = imgname
                if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): #dataset contains tuples of (img,label)
                    imglabel = img_to_open[1]
                    img_to_open = img_to_open[0]
                # TODO: This is problematic, as the maximum prototype might not relate to the class that was predicted
                imglabel = projectloader.dataset.class_idx_to_name(pred_class)

                image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_to_open).convert("RGB"))
                img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, args.wshape, patchsize, skip, max_h_idx_class, max_w_idx_class)
                img_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                saved[pf_index]+=1
                prototype_patches[pf_index].append((img_patch, max_class_patch_pf))
                
                save_path = os.path.join(dir, f"prototype_{pf_index}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                draw = D.Draw(image)
                draw.rectangle([(w_coor_min,h_coor_min), (w_coor_max, h_coor_max)], outline='yellow', width=2)
                image.save(os.path.join(save_path, f"p{pf_index}_{imglabel}_{round(max_class_patch_pf, 2)}_{imgname.split('/')[-1].split('.jpg')[0]}_rect.png"))

    print("num images abstained: ", len(abstainedimgs))
    print("num images not abstained: ", len(notabstainedimgs))
    for p in range(net.module._num_prototypes):
        if saved[p]>0:
            try:
                sorted_by_second = sorted(prototype_patches[p], key=lambda tup: tup[1], reverse=True)
                sorted_ps = [i[0] for i in sorted_by_second]
                grid = torchvision.utils.make_grid(sorted_ps, nrow=16, padding=1)
                torchvision.utils.save_image(grid,os.path.join(dir,"grid_%s.png"%(str(p))))
            except RuntimeError:
                pass

def visualize(net, projectloader, num_classes, device, foldername, args: argparse.Namespace):
    if args.task == "classification":
        return visualize_cls(net, projectloader, num_classes, device, foldername, args)
    elif args.task == "segmentation":
        return visualize_seg(net, projectloader, num_classes, device, foldername, args)

# convert latent location to coordinates of image patch
def get_img_coordinates(img_size, wshape, patchsize, skip, h_idx, w_idx):
    # in case latent output size is 26x26. For convnext with smaller strides. 
    if wshape == 26 and wshape == 26:
        #Since the outer latent patches have a smaller receptive field, skip size is set to 4 for the first and last patch. 8 for rest.
        h_coor_min = max(0, round((h_idx-1) * skip + 4))
        if h_idx < wshape-1:
            h_coor_max = h_coor_min + patchsize
        else:
            h_coor_min -= 4
            h_coor_max = h_coor_min + patchsize
        w_coor_min = max(0, round((w_idx-1) * skip + 4))
        if w_idx < wshape-1:
            w_coor_max = w_coor_min + patchsize
        else:
            w_coor_min -= 4
            w_coor_max = w_coor_min + patchsize
    else:
        h_coor_min = round(h_idx * skip)
        h_coor_max = min(img_size, round(h_idx*skip + patchsize))
        w_coor_min = round(w_idx * skip)
        w_coor_max = min(img_size, round(w_idx * skip+patchsize))

    return h_coor_min, h_coor_max, w_coor_min, w_coor_max



@torch.no_grad()
def topk_images(args, net, projectloader, device, use_precomp_topk=True):
    net = net.to(device)
    net.eval()

    topk_json_file_path = os.path.join(args.log_dir, "topk_classes.json")
    if os.path.exists(topk_json_file_path) and use_precomp_topk:
        print("Reading in topk images from file")
        with open(topk_json_file_path) as f:
            topk_dict = json.load(f)
        # convert keys that were read in as str back to int
        topk_dict = {int(key): value for key, value in topk_dict.items()}

    else:
        outs = []
        net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) 
        for xs, ys in tqdm(projectloader, ncols=0):
            xs, ys = xs.to(device), ys.to(device)
            
            with torch.no_grad():
                _, _, _, _, out = net(xs, inference=True)
                outs.append(out)
            del out
        outs = torch.cat(outs)

        topk_classes = outs.topk(k=10, dim=0)[1].T
        
        topk_dict = {}
        for i in range(topk_classes.shape[0]):
            topk_dict[i] = topk_classes[i].tolist()
    
        topk_json = json.dumps(topk_dict, indent=4, cls=NpEncoder)
        with open(topk_json_file_path, "w") as f:
            f.write(topk_json)
    
    return topk_dict
    

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)